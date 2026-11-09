/**
 * Line editor for String List and Multiline Text (Advanced).
 *
 * A view onto the `lines` widget: each line in its own colour with a switch, every edit written
 * back. A wire into `text` sets the lines aside.
 */

import { app } from "../../scripts/app.js";
import { CARD_LOOK, cardChip, cardRule } from "./interface/card.js";
import { captureWheel, wheelPixels } from "./interface/pointer.js";
import { BLANK, NOTE, OFF, OFF_MARK, PROMPT, classify, handedOn, joinLines } from "./interface/prompt_lines.js";
import { withGraphChange } from "./interface/region.js";
import { ROW_PALETTE } from "./interface/row_headers.js";
import { themeVar } from "./interface/theme.js";
import { setWidgetHidden } from "./interface/visibility.js";
import { appendInterfaceWidget, chainWidgetCallback } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.LineEditorUI";
const SETTING_ID = "WAS.LineEditor.ShowInterface";

const LINES_WIDGET = "lines";
const WIRE_INPUT = "text";
const BLANK_WIDGET = "skip_blank_lines";
const COMMENT_WIDGET = "skip_comment_lines";
const JOIN_WIDGET = "join_with";

// The nodes this draws on: what a kept line is called, the schema's blank-line default, and
// whether kept lines are numbered or joined into one text.
const NODES = {
  WASStringList: { one: "entry", many: "entries", skipBlank: true, numbered: true, joined: false },
  WASMultilineTextAdvanced: { one: "line kept", many: "lines kept", skipBlank: false, numbered: false, joined: true },
};

const UI_WIDGET_NAME = "was_line_editor_ui";
const UI_WIDGET_TYPE = "was_line_editor";

// Characters of the joined text shown on hover.
const PREVIEW_CHARS = 2000;

// The height the panel opens at and the width its node opens at, in node units.
const PANEL_HEIGHT = 240;
const PANEL_WIDTH = 360;

// Layout, in CSS pixels.
const GUTTER = 50;
const PAD_X = 8;
const PAD_Y = 6;
const LINE_HEIGHT = 18;
const STRIPE = 3;
const BOX = 13;

// The kind of the empty line after a final line break, which is drawn and never handed on.
const END = "end";

const FG = themeVar("fg");
const MUTED = themeVar("fgMuted");
const DISABLED = themeVar("fgDisabled");
const ACCENT = themeVar("accent");
const DEEP = themeVar("bgDark");

// The type the text box and the lines under it both take.
const TEXT_LOOK = "font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;font-size:12px;"
  + `font-weight:400;font-style:normal;line-height:${LINE_HEIGHT}px;letter-spacing:normal;`
  + "word-spacing:normal;text-indent:0;text-transform:none;tab-size:4;white-space:pre-wrap;"
  + "overflow-wrap:break-word;word-break:normal;font-kerning:auto;font-variant-ligatures:normal;";

// The colours a prompt line cycles through, from the row header palette.
const PROMPT_HUES = ["green", "yellow", "orange", "pink", "purple", "red"]
  .map((name) => ROW_PALETTE[name]?.stripe)
  .filter(Boolean);

// What the panel says over an empty list.
const HINT = "One prompt per line · ~ switches a line off · # or // starts a note";

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
 * @param {boolean} fallback - The schema's default.
 * @returns {boolean} The widget's value, or the default when it holds no boolean.
 */
function readSwitch(node, name, fallback) {
  const value = findWidget(node, name)?.value;
  return typeof value === "boolean" ? value : fallback;
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
 * Read whether the editor is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function interfaceEnabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID)
      ?? app?.ui?.settings?.getSettingValue?.(SETTING_ID);
    return typeof value === "boolean" ? value : true;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
    return true;
  }
}

/**
 * Fold the `lines` box away, the editor standing in for it.
 *
 * @param {object} node - The String List node.
 * @returns {void}
 */
function foldTextBox(node) {
  if (setWidgetHidden(findWidget(node, LINES_WIDGET), true)) node.setDirtyCanvas?.(true, true);
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

/**
 * The fill, stripe and text colour of one line.
 *
 * @param {string} kind - What the line is.
 * @param {number} hue - Its place among the prompt lines, or -1.
 * @param {boolean} handed - Whether it becomes an entry.
 * @returns {{fill: string, stripe: string, text: string, strike: boolean}} Its look.
 */
function lineLook(kind, hue, handed) {
  if (kind === PROMPT) {
    const colour = PROMPT_HUES[hue % PROMPT_HUES.length] ?? ACCENT;
    return { fill: tint(colour, 13), stripe: colour, text: FG, strike: false };
  }
  if (kind === OFF) return { fill: tint(FG, 3), stripe: DISABLED, text: DISABLED, strike: true };
  if (kind === NOTE) {
    return { fill: tint(ACCENT, 15), stripe: ACCENT, text: `color-mix(in srgb, ${ACCENT} 55%, ${FG})`, strike: false };
  }
  if (kind === BLANK && handed) return { fill: tint(FG, 4), stripe: tint(FG, 18), text: FG, strike: false };
  return { fill: "transparent", stripe: "transparent", text: FG, strike: false };
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
  gutter.style.cssText = `flex:0 0 ${GUTTER}px;display:flex;align-items:flex-start;gap:4px;`
    + `box-sizing:border-box;padding:${(LINE_HEIGHT - BOX) / 2}px 0 0 ${STRIPE + 5}px`;
  const box = document.createElement("input");
  box.type = "checkbox";
  box.tabIndex = -1;
  box.style.cssText = `flex:none;margin:0;width:${BOX}px;height:${BOX}px;cursor:pointer;`
    + `accent-color:${ACCENT}`;
  const glyph = document.createElement("span");
  glyph.style.cssText = `flex:none;width:${BOX}px;text-align:center;`
    + `font:600 11px/${BOX}px system-ui,sans-serif;color:${ACCENT}`;
  const number = document.createElement("span");
  number.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-align:right;"
    + `padding-right:5px;font:500 10px/${BOX}px system-ui,sans-serif;`
    + `font-variant-numeric:tabular-nums;color:${MUTED}`;
  gutter.append(box, glyph, number);
  const body = document.createElement("div");
  body.style.cssText = `flex:1 1 auto;min-width:0;box-sizing:border-box;padding:0 ${PAD_X}px;${TEXT_LOOK}`;
  row.append(gutter, body);
  return { row, box, glyph, number, body, key: "" };
}

/**
 * Build the editor for one node.
 *
 * @param {object} node - The node it draws.
 * @param {object} setup - Its entry in `NODES`.
 * @returns {object} The panel: its element and height, and `sync`, `refresh` and `dispose`.
 */
function createEditor(node, setup) {
  const textWidget = () => findWidget(node, LINES_WIDGET);
  const linked = () => inputLinked(node, WIRE_INPUT);

  const root = document.createElement("div");
  root.style.cssText = "display:flex;flex-direction:column;width:100%;height:100%;min-height:0;"
    + "box-sizing:border-box;overflow:hidden";

  const head = document.createElement("div");
  head.style.cssText = "flex:none;display:flex;align-items:center;gap:6px;min-width:0;padding:6px 8px 4px";
  const summary = document.createElement("div");
  summary.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + `white-space:nowrap;font:11.5px/16px system-ui,sans-serif;color:${CARD_LOOK.muted}`;
  const chips = document.createElement("div");
  chips.style.cssText = "flex:none;display:flex;gap:4px";
  head.append(summary, chips);

  const scroller = document.createElement("div");
  scroller.style.cssText = "flex:1 1 auto;min-height:0;overflow:auto;position:relative;"
    + `scrollbar-width:thin;scrollbar-color:${themeVar("scrollbarThumb")} transparent`;
  const sheet = document.createElement("div");
  sheet.style.cssText = "position:relative;min-height:100%;box-sizing:border-box";
  const lines = document.createElement("div");
  lines.style.cssText = `position:relative;padding:${PAD_Y}px 0`;
  const hint = document.createElement("div");
  hint.style.cssText = `position:absolute;top:${PAD_Y}px;left:${GUTTER + PAD_X}px;right:${PAD_X}px;`
    + `pointer-events:none;color:${MUTED};${TEXT_LOOK}`;
  hint.textContent = HINT;
  const box = document.createElement("textarea");
  box.spellcheck = false;
  box.setAttribute("aria-label", "Lines");
  box.style.cssText = `position:absolute;top:0;left:${GUTTER}px;width:calc(100% - ${GUTTER}px);height:100%;`
    + "margin:0;border:0;outline:none;resize:none;overflow:hidden;box-sizing:border-box;"
    + `padding:${PAD_Y}px ${PAD_X}px;background:transparent;color:transparent;`
    + `-webkit-text-fill-color:transparent;caret-color:${FG};${TEXT_LOOK}`;
  sheet.append(lines, hint, box);
  scroller.append(sheet);
  root.append(head, cardRule(), scroller);

  const rows = [];
  let current = -1;
  let frame = 0;

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
   * Draw one line's row from what it is.
   *
   * @param {object} row - The row's parts.
   * @param {object} line - `{kind, text, number, hue, handed, current, switchable, off}`.
   * @param {boolean} wired - Whether the typed lines are set aside for a wire.
   * @returns {void}
   */
  const paintRow = (row, line, wired) => {
    const key = JSON.stringify([line, wired]);
    if (row.key === key) return;
    row.key = key;
    const look = lineLook(line.kind, line.hue, line.handed);
    const ring = line.current ? `, inset 0 0 0 1px ${tint(ACCENT, 75)}` : "";
    const lift = line.current ? `linear-gradient(${tint(FG, 6)}, ${tint(FG, 6)}), ` : "";
    row.row.style.background = `${lift}${look.fill}`;
    row.row.style.boxShadow = `inset ${STRIPE}px 0 0 ${look.stripe}, `
      + `inset 0 -1px 0 ${tint(DEEP, 55)}${ring}`;
    row.body.textContent = line.text === "" ? "​" : line.text;
    row.body.style.color = look.text;
    row.body.style.textDecoration = look.strike ? `line-through ${tint(FG, 45)}` : "none";
    row.box.style.display = line.switchable ? "" : "none";
    row.box.checked = !line.off;
    row.box.disabled = wired;
    row.box.title = line.off ? "Switch this line on" : "Switch this line off";
    row.glyph.style.display = !line.switchable && line.kind === NOTE ? "" : "none";
    row.glyph.textContent = "i";
    row.glyph.title = "Note, left out";
    row.number.textContent = line.number ? String(line.number) : "";
    row.number.title = line.number ? `Entry ${line.number} of the list` : "";
  };

  /**
   * Write the summary over the lines.
   *
   * @param {object} counts - `{entries, off, notes, text, wired, joined}`.
   * @returns {void}
   */
  const paintSummary = ({ entries, off, notes, text, wired, joined }) => {
    summary.replaceChildren();
    chips.replaceChildren();
    summary.title = "";
    const strong = (value) => {
      const figure = document.createElement("span");
      figure.style.cssText = `font-weight:650;color:${CARD_LOOK.title}`;
      figure.textContent = String(value);
      return figure;
    };
    if (wired) {
      summary.append("Using the wired text");
      chips.append(cardChip("wired", "accent"));
      return;
    }
    if (!text) {
      summary.append("Nothing typed yet");
      return;
    }
    summary.append(strong(entries), ` ${entries === 1 ? setup.one : setup.many}`);
    if (off) summary.append(" · ", strong(off), " off");
    if (notes) summary.append(" · ", strong(notes), notes === 1 ? " note" : " notes");
    if (setup.joined) {
      summary.append(" · ", strong(joined.length), joined.length === 1 ? " character" : " characters");
      summary.title = joined.length > PREVIEW_CHARS ? `${joined.slice(0, PREVIEW_CHARS)}…` : joined;
    }
    if (!entries) chips.append(cardChip(setup.joined ? "empty text" : "nothing to hand on", "warning"));
  };

  /**
   * Draw every line from the text box and the two switches.
   *
   * @returns {void}
   */
  const render = () => {
    cancelAnimationFrame(frame);
    frame = 0;
    foldTextBox(node);
    const wired = linked();
    const text = box.value;
    const skipBlank = readSwitch(node, BLANK_WIDGET, setup.skipBlank);
    const skipComments = readSwitch(node, COMMENT_WIDGET, true);
    const kinds = classify(text);
    const raw = text.split("\n");
    current = caretLine();
    while (rows.length < raw.length) {
      const row = createRow();
      const index = rows.length;
      row.box.addEventListener("pointerdown", (event) => event.preventDefault());
      row.box.addEventListener("click", (event) => {
        event.stopPropagation();
        toggle(index);
      });
      rows.push(row);
      lines.append(row.row);
    }
    while (rows.length > raw.length) rows.pop().row.remove();

    let entries = 0;
    let hue = 0;
    let off = 0;
    let notes = 0;
    raw.forEach((value, index) => {
      const known = kinds[index];
      const kind = known?.kind ?? END;
      const handed = kind !== END && handedOn(kind, skipBlank, skipComments);
      if (handed && !(setup.joined && kind === BLANK)) entries += 1;
      if (kind === OFF) off += 1;
      if (kind === NOTE && !handed) notes += 1;
      const prompt = kind === PROMPT || kind === OFF;
      paintRow(rows[index], {
        kind,
        text: value,
        number: handed && !wired && setup.numbered ? entries : 0,
        hue: prompt ? hue : -1,
        handed,
        current: index === current,
        switchable: prompt || (kind === NOTE && !skipComments && !known?.block),
        off: kind === OFF,
      }, wired);
      if (prompt) hue += 1;
    });

    hint.style.display = text ? "none" : "";
    box.readOnly = wired;
    sheet.style.opacity = wired ? "0.55" : "1";
    const joined = setup.joined
      ? joinLines(text, skipBlank, skipComments, String(findWidget(node, JOIN_WIDGET)?.value ?? "line break"))
      : "";
    paintSummary({ entries, off, notes, text, wired, joined });
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
   * Scroll the caret's line into view.
   *
   * @returns {void}
   */
  const revealCaret = () => {
    const row = rows[current]?.row;
    if (!row) return;
    const top = row.offsetTop;
    const bottom = top + row.offsetHeight;
    if (top < scroller.scrollTop) scroller.scrollTop = top;
    else if (bottom > scroller.scrollTop + scroller.clientHeight) {
      scroller.scrollTop = Math.min(top, bottom - scroller.clientHeight);
    }
  };

  /**
   * Write the text box into the widget, then draw.
   *
   * @returns {void}
   */
  const commit = () => {
    const widget = textWidget();
    if (widget && box.value !== normalise(widget.value)) {
      widget.value = box.value;
      node.setDirtyCanvas?.(true, true);
    }
    render();
    revealCaret();
  };

  /**
   * Switch one line on or off by its leading `~`, as one edit of the text box.
   *
   * @param {number} index - Line index from zero.
   * @returns {void}
   */
  const toggle = (index) => {
    if (linked()) return;
    const text = box.value;
    const all = text.split("\n");
    if (index >= all.length) return;
    let at = 0;
    for (let line = 0; line < index; line += 1) at += all[line].length + 1;
    const line = all[index];
    const lead = line.length - line.replace(/^\s+/u, "").length;
    at += lead;
    const off = line.slice(lead).startsWith(OFF_MARK);
    // The caret stays where it was, or goes to the end of this line when the box had no focus.
    const lineEnd = at - lead + line.length;
    const focused = document.activeElement === box;
    const [start, end] = focused ? [box.selectionStart ?? 0, box.selectionEnd ?? 0] : [lineEnd, lineEnd];
    const moved = (position) => {
      if (off) return position > at ? position - 1 : position;
      return position >= at ? position + 1 : position;
    };
    withGraphChange(() => {
      box.focus({ preventScroll: true });
      box.setSelectionRange(at, off ? at + 1 : at);
      let done = false;
      try {
        // Keeps the text box's own undo.
        done = off
          ? document.execCommand("delete", false)
          : document.execCommand("insertText", false, OFF_MARK);
      } catch {
        done = false;
      }
      if (!done) {
        box.value = off ? text.slice(0, at) + text.slice(at + 1) : text.slice(0, at) + OFF_MARK + text.slice(at);
        commit();
      }
      box.setSelectionRange(moved(start), moved(end));
    });
    render();
  };

  /**
   * Load the widget's text into the box where something else changed it, then draw.
   *
   * @returns {void}
   */
  const sync = () => {
    const value = normalise(textWidget()?.value);
    if (value !== box.value) box.value = value;
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

  // A press under the last line puts the caret at the end.
  scroller.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || (event.target !== scroller && event.target !== sheet)) return;
    event.preventDefault();
    box.focus({ preventScroll: true });
    box.setSelectionRange(box.value.length, box.value.length);
  });
  root.addEventListener("pointerdown", (event) => {
    if (event.button === 1) app.canvas?.processMouseDown?.(event);
  });
  head.addEventListener("contextmenu", (event) => event.preventDefault());
  lines.addEventListener("contextmenu", (event) => event.preventDefault());
  const releaseWheel = captureWheel(root, (event) => {
    if (scroller.scrollHeight <= scroller.clientHeight) return false;
    const before = scroller.scrollTop;
    scroller.scrollTop += wheelPixels(event, scroller).y;
    return scroller.scrollTop !== before;
  });

  return {
    element: root,
    height: PANEL_HEIGHT,
    minWidth: PANEL_WIDTH,
    sync,
    schedule,
    refresh: sync,
    dispose() {
      cancelAnimationFrame(frame);
      document.removeEventListener("selectionchange", onSelection);
      releaseWheel();
    },
  };
}

/**
 * Put the editor on a node and wire it to the widgets it reads.
 *
 * @param {object} node - The node being created.
 * @param {object} setup - Its entry in `NODES`.
 * @returns {void}
 */
function attachEditor(node, setup) {
  if (!findWidget(node, LINES_WIDGET)) return;
  const panel = createEditor(node, setup);
  foldTextBox(node);
  appendInterfaceWidget(node, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  chainWidgetCallback(node, LINES_WIDGET, panel.sync, EXT_NAME);
  for (const name of [BLANK_WIDGET, COMMENT_WIDGET, JOIN_WIDGET]) {
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
        "Draw the lines of String List and Multiline Text (Advanced) in colour, each with a "
        + "switch. Off, the plain text box is drawn instead and works the same way. This applies "
        + "to nodes added after the setting changes, so a reload shows it everywhere.",
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
        if (interfaceEnabled()) attachEditor(this, setup);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the line editor:`, error);
      }
      return result;
    };
  },
});
