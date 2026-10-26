/**
 * A chat drawn on a node: a menu of saved conversations, the conversation as bubbles, the reply
 * streaming in, and a message box that queues.
 *
 * Reads the run result body `transcript`, JSON lines of `{role, text, pictures}`, and
 * `/was/interface/api/llm_sessions`.
 */

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { captureWheel, wheelPixels } from "./pointer.js";
import { openPopupMenu } from "./popup_menu.js";
import { executionId } from "./preview.js";
import { fetchWithin } from "./request.js";
import { onNodeFinished, onRunEnded } from "./run_events.js";
import { fetchRunResult, fetchRunResultPage, whenReadable } from "./run_result.js";
import { inputValue, sourceOf } from "./upstream.js";
import { setWidgetHidden } from "./visibility.js";
import { FIELD, LOOK, SANS, SCROLLBARS, button, el, focusRing, pill, setDisabled } from "./window_kit.js";

const LOG_NAME = "WASNodeSuite.ChatPanel";

const ROUTE = "/was/interface/api/llm_sessions";

// The body of the run result holding the conversation.
const TRANSCRIPT_BODY = "transcript";

// Lines of the conversation asked for at once.
const TRANSCRIPT_LINES = 500;

// Height in node units the panel opens at, and the narrowest it is drawn in.
const PANEL_HEIGHT = 340;
const MIN_WIDTH = 320;

// Characters of one bubble drawn before it is cut.
const BUBBLE_CHARS = 20000;

// Characters of a first message a new conversation's name is cut from.
const NAME_CHARS = 40;

// What the session fact says for a conversation kept for one run.
const ONE_RUN = "this run only";

// The node whose `session` names the conversation when none is chosen here.
const MEMORY_NODE = "WASLLMMemory";

// A saved name: letters, digits, spaces, dots, dashes and underscores, up to 64 characters.
const NAME = /^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$/;

// Names Windows opens as a device in every folder.
const DEVICES = new Set(["CON", "PRN", "AUX", "NUL",
  ...Array.from({ length: 10 }, (_, d) => `COM${d}`), ...Array.from({ length: 10 }, (_, d) => `LPT${d}`)]);

/**
 * The widget a node carries under a name.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @returns {object|undefined} The widget.
 */
function widgetNamed(node, name) {
  return (node.widgets ?? []).find((widget) => widget?.name === name);
}

/**
 * Whether an input is wired from another node.
 *
 * @param {object} node - The node.
 * @param {string} name - The input's or its widget's name.
 * @returns {boolean} True when a link feeds it.
 */
function wired(node, name) {
  const input = (node.inputs ?? []).find((slot) => slot?.widget?.name === name || slot?.name === name);
  return input?.link != null;
}

/**
 * Set a widget's value the way a person typing it would.
 *
 * @param {object} node - The node.
 * @param {object} widget - The widget.
 * @param {*} value - The new value.
 * @returns {void}
 */
function setWidget(node, widget, value) {
  if (!widget) return;
  widget.value = value;
  try {
    widget.callback?.(value, app.canvas, node);
  } catch (error) {
    console.error(`[${LOG_NAME}] ${widget.name} did not take the new value:`, error);
  }
  node.setDirtyCanvas?.(true, true);
}

/**
 * The session name the LLM Memory wired into an input holds, read off the page.
 *
 * @param {object} node - The node.
 * @param {string} name - The memory input's name.
 * @returns {string|null} The name, empty when unwired or unnamed, null when only a run can tell.
 */
function memorySession(node, name) {
  if (!wired(node, name)) return "";
  const source = sourceOf(node, name);
  if (source?.node?.type !== MEMORY_NODE) return null;
  const held = inputValue(source.node, "session");
  return held.known && typeof held.value === "string" ? held.value.trim() : null;
}

/**
 * Whether a name can be saved under.
 *
 * @param {string} name - The name.
 * @returns {boolean} True when the server takes it.
 */
function usableName(name) {
  if (!NAME.test(name) || /[. ]$/.test(name)) return false;
  return !DEVICES.has(name.split(".", 1)[0].trim().toUpperCase());
}

/**
 * A name for a new conversation, from its first message, unlike any taken.
 *
 * @param {string} text - The first message.
 * @param {Set<string>} taken - Saved names, lower case.
 * @returns {string} The name.
 */
function nameFrom(text, taken) {
  let base = String(text ?? "").replace(/[^A-Za-z0-9 ._-]+/g, " ").replace(/\s+/g, " ").trim();
  base = base.replace(/^[^A-Za-z0-9]+/, "");
  if (base.length > NAME_CHARS) {
    const cut = base.slice(0, NAME_CHARS);
    const space = cut.lastIndexOf(" ");
    base = space > NAME_CHARS / 2 ? cut.slice(0, space) : cut;
  }
  base = base.replace(/[ .]+$/, "");
  if (!usableName(base)) {
    const now = new Date();
    const two = (value) => String(value).padStart(2, "0");
    base = `Chat ${now.getFullYear()}-${two(now.getMonth() + 1)}-${two(now.getDate())} `
      + `${two(now.getHours())}.${two(now.getMinutes())}`;
  }
  let name = base;
  for (let n = 2; taken.has(name.toLowerCase()); n += 1) name = `${base} ${n}`;
  return name;
}

/**
 * The messages of a transcript, oldest first.
 *
 * @param {string[]} lines - JSON lines.
 * @returns {Array<{role: string, text: string, pictures: number}>} The messages that parse.
 */
function parseTranscript(lines) {
  const messages = [];
  for (const line of lines ?? []) {
    if (!line || !line.trim()) continue;
    try {
      const row = JSON.parse(line);
      if (!row || typeof row !== "object") continue;
      messages.push({
        role: String(row.role ?? "assistant"),
        text: String(row.text ?? ""),
        pictures: Number.isFinite(row.pictures) ? row.pictures : 0,
      });
    } catch {
      // A line cut by the page limit is left out.
    }
  }
  return messages;
}

/**
 * One message, drawn as a bubble.
 *
 * @param {{role: string, text: string, pictures: number}} message - The message.
 * @returns {{row: HTMLElement, body: HTMLElement}} The bubble's row and its body.
 */
function bubble(message) {
  const mine = message.role === "user";
  const quiet = message.role === "system" || message.role === "tool";
  const row = el("div", `display:flex;justify-content:${mine ? "flex-end" : "flex-start"}`);
  const body = el("div", [
    "max-width:86%",
    "padding:7px 10px",
    "border-radius:10px",
    mine ? "border-bottom-right-radius:3px" : "border-bottom-left-radius:3px",
    `font:${quiet ? "11px" : "12.5px"}/1.5 ${SANS}`,
    "white-space:pre-wrap",
    "overflow-wrap:anywhere",
    `color:${quiet ? LOOK.muted : LOOK.text}`,
    mine
      ? `background:color-mix(in srgb, ${LOOK.accent} 26%, ${LOOK.surface})`
      : `background:${LOOK.surface}`,
    `border:1px solid ${mine ? `color-mix(in srgb, ${LOOK.accent} 45%, ${LOOK.border})` : LOOK.border}`,
  ].join(";"));
  if (quiet) body.appendChild(el("div", `font:600 10px ${SANS};text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px`, message.role));
  if (message.pictures > 0) {
    body.appendChild(pill(`${message.pictures} picture${message.pictures === 1 ? "" : "s"}`));
    if (message.text) body.appendChild(el("div", "height:4px"));
  }
  const text = message.text.length > BUBBLE_CHARS ? `${message.text.slice(0, BUBBLE_CHARS)}...` : message.text;
  body.appendChild(document.createTextNode(text));
  row.appendChild(body);
  return { row, body };
}

/**
 * Ask the conversations route, answering its JSON or `null`.
 *
 * @param {string} query - The query string, `""` for the listing.
 * @param {string} [method] - `GET` or `POST`.
 * @returns {Promise<{status: number, body: object|null}>} What it answered.
 */
async function ask(query, method = "GET") {
  try {
    const answer = await fetchWithin(`${ROUTE}${query ? `?${query}` : ""}`, { method, cache: "no-store" });
    if (answer.status === 204) return { status: 204, body: null };
    const body = await answer.json().catch(() => null);
    return { status: answer.status, body };
  } catch (error) {
    console.error(`[${LOG_NAME}] ${ROUTE} did not answer:`, error);
    return { status: 0, body: null };
  }
}

/**
 * Build the chat panel for a node.
 *
 * @param {object} node - The node, carrying `message`, `reset` and `conversation` widgets.
 * @param {object} [options] - `{messageWidget, resetWidget, conversationWidget, memoryInput,
 *   height}`.
 * @returns {{element: HTMLElement, height: number, minWidth: number, refresh: Function,
 *   dispose: Function}} The panel, for `appendInterfaceWidget`.
 */
export function createChatPanel(node, options = {}) {
  const messageName = options.messageWidget ?? "message";
  const resetName = options.resetWidget ?? "reset";
  const conversationName = options.conversationWidget ?? "conversation";
  const memoryName = options.memoryInput ?? "memory";
  const height = Number(options.height) > 0 ? Number(options.height) : PANEL_HEIGHT;

  const root = el("div", [
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    "display:flex",
    "flex-direction:column",
    "gap:6px",
    "padding:8px",
    "border-radius:6px",
    `background:${LOOK.body}`,
    `border:1px solid ${LOOK.border}`,
    `font:12px ${SANS}`,
    `color:${LOOK.text}`,
  ].join(";"));
  root.className = "was-llm-chat";

  const header = el("div", "display:flex;align-items:center;gap:6px;flex:0 0 auto;min-width:0");
  const picker = button("", () => openPicker(), { kind: "tool", hint: "Open a saved conversation", logName: LOG_NAME });
  Object.assign(picker.style, { flex: "0 1 auto", minWidth: "0", justifyContent: "flex-start", maxWidth: "100%" });
  const pickerName = el("span", "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0");
  const caret = el("span", `flex:0 0 auto;color:${LOOK.muted};font-size:10px`, "▾");
  picker.append(pickerName, caret);
  const renameBox = focusRing(el("input", `${FIELD};flex:1 1 auto;padding:3px 8px;font-size:12px;display:none`));
  renameBox.maxLength = 64;
  renameBox.spellcheck = false;
  const model = pill("");
  const count = pill("0 messages");
  const resetOn = pill("reset on", LOOK.warning);
  resetOn.title = "deleted before each run";
  for (const fact of [model, count]) {
    Object.assign(fact.style, { flex: "0 1 auto", minWidth: "0", overflow: "hidden", textOverflow: "ellipsis", display: "block" });
  }
  model.style.display = "none";
  resetOn.style.display = "none";
  const spacer = el("div", "flex:1 1 auto;min-width:4px");
  const newChat = button("New", () => startNew(), { kind: "ghost", hint: "Start a new conversation", logName: LOG_NAME });
  const more = button("⋯", (event) => openActions(event), { kind: "ghost", hint: "Rename or delete this conversation", logName: LOG_NAME });
  header.append(picker, renameBox, model, count, resetOn, spacer, newChat, more);

  const notice = el("div", `display:none;flex:0 0 auto;align-items:center;gap:6px;padding:5px 8px;border-radius:6px;`
    + `font:12px/1.4 ${SANS};border:1px solid ${LOOK.border};background:${LOOK.surface}`);

  const transcript = el("div", `flex:1 1 auto;min-height:0;overflow:auto;display:flex;`
    + `flex-direction:column;gap:6px;padding:2px;${SCROLLBARS}`);
  const empty = el("div", `margin:auto;color:${LOOK.muted};font:12px ${SANS}`, "send a message to start");
  transcript.appendChild(empty);

  const composer = el("div", "display:flex;gap:6px;align-items:flex-end;flex:0 0 auto");
  const box = focusRing(el("textarea", [
    "box-sizing:border-box",
    "flex:1 1 auto",
    "min-width:0",
    "resize:none",
    "padding:7px 9px",
    "border-radius:8px",
    `font:12.5px/1.45 ${SANS}`,
    `color:${LOOK.text}`,
    `background:${LOOK.surface}`,
    `border:1px solid ${LOOK.border}`,
    "outline:none",
    SCROLLBARS,
  ].join(";")));
  box.rows = 2;
  box.placeholder = "Message";
  box.spellcheck = true;
  const send = button("Send", () => submit(), { kind: "primary", hint: "Send and run the graph (Enter)", logName: LOG_NAME });
  composer.append(box, send);
  root.append(header, notice, transcript, composer);

  let disposed = false;
  let pending = null;
  let sentText = "";
  let drawnKey = "";
  let lastSession = "";
  // True after New, until the first message names the conversation.
  let fresh = false;
  // Counts loads, so an older answer arriving late is dropped.
  let loads = 0;

  const conversationWidget = () => widgetNamed(node, conversationName);
  const chosen = () => String(conversationWidget()?.value ?? "").trim();
  const current = () => chosen() || (fresh ? "" : lastSession || memorySession(node, memoryName) || "");

  const say = (text, tone) => {
    notice.textContent = "";
    if (!text) {
      notice.style.display = "none";
      return;
    }
    notice.appendChild(el("span", `flex:1 1 auto;color:${tone ?? LOOK.text}`, text));
    notice.style.display = "flex";
  };

  const showHeader = () => {
    const name = current();
    const fromMemory = !chosen() && !fresh && wired(node, memoryName) && name;
    const unknown = !fresh && memorySession(node, memoryName) === null;
    pickerName.textContent = name || (unknown ? "LLM Memory's session" : "New conversation");
    picker.title = fromMemory ? `${name}, named by LLM Memory` : "Open a saved conversation";
    const locked = wired(node, conversationName);
    setDisabled(picker, locked);
    setDisabled(newChat, locked);
    setDisabled(more, locked || !name);
    resetOn.style.display = widgetNamed(node, resetName)?.value ? "" : "none";
  };

  const showWiring = () => {
    const fromLink = wired(node, messageName);
    box.disabled = fromLink;
    box.placeholder = fromLink ? "message is wired from another node" : "Message";
    box.style.opacity = fromLink ? "0.6" : "1";
  };

  const scrollToEnd = () => {
    transcript.scrollTop = transcript.scrollHeight;
  };

  const draw = (messages, facts = {}) => {
    const key = JSON.stringify([current(), messages.length, messages.at(-1)?.text?.length ?? 0, messages.at(-1)?.role]);
    if (facts.model) {
      model.textContent = String(facts.model);
      model.style.display = "";
    }
    count.textContent = `${messages.length} message${messages.length === 1 ? "" : "s"}`;
    showHeader();
    if (key === drawnKey && !pending) return;
    drawnKey = key;
    pending = null;
    transcript.textContent = "";
    if (!messages.length) {
      transcript.appendChild(empty);
      return;
    }
    for (const message of messages) transcript.appendChild(bubble(message).row);
    scrollToEnd();
  };

  // A saved conversation's transcript, read without running the graph.
  const loadSaved = async (name) => {
    const ticket = ++loads;
    const answer = await ask(`name=${encodeURIComponent(name)}`);
    if (disposed || ticket !== loads) return;
    const lines = typeof answer.body?.transcript === "string" ? answer.body.transcript.split("\n") : [];
    draw(parseTranscript(lines));
  };

  const refresh = async () => {
    if (disposed) return;
    showHeader();
    showWiring();
    const ticket = ++loads;
    const answer = await fetchRunResult(node);
    if (disposed || ticket !== loads) return;
    const report = answer?.result;
    const facts = {};
    for (const { name, value } of report?.facts ?? []) facts[name] = value;
    const ran = String(facts.session ?? "");
    if (report) lastSession = ran && ran !== ONE_RUN ? ran : "";
    const wanted = chosen() || (fresh ? "" : memorySession(node, memoryName) || "");
    if (fresh && !wanted) {
      draw([], facts);
      return;
    }
    // The conversation picked since the last run is read from where it is saved.
    if (wanted && wanted !== ran) {
      if (facts.model) model.textContent = String(facts.model);
      await loadSaved(wanted);
      return;
    }
    const index = (report?.bodies ?? []).findIndex((body) => body?.name === TRANSCRIPT_BODY);
    if (index < 0) {
      draw([], facts);
      return;
    }
    const page = await fetchRunResultPage(node, index, 0, TRANSCRIPT_LINES);
    if (disposed || ticket !== loads) return;
    draw(parseTranscript(page?.page?.lines ?? []), facts);
  };

  const savedRows = async () => {
    const answer = await ask("");
    return Array.isArray(answer.body?.sessions) ? answer.body.sessions : [];
  };

  const choose = (name) => {
    say("");
    fresh = false;
    setWidget(node, conversationWidget(), name);
    drawnKey = "";
    showHeader();
    loadSaved(name);
  };

  const startNew = () => {
    say("");
    fresh = true;
    setWidget(node, conversationWidget(), "");
    drawnKey = "";
    draw([]);
    box.focus({ preventScroll: true });
  };

  const openPicker = async () => {
    const rect = picker.getBoundingClientRect();
    const rows = await savedRows();
    if (disposed) return;
    const name = current();
    const items = [{ header: rows.length ? "Saved conversations" : "No saved conversations yet" }];
    for (const row of rows) {
      const held = Number.isFinite(row.messages) ? `${row.messages} msg · ` : "";
      items.push({
        label: String(row.name),
        detail: `${held}${String(row.updated ?? "")}`,
        checked: String(row.name) === name,
        onSelect: () => choose(String(row.name)),
      });
    }
    items.push({ separator: true }, { label: "New conversation", onSelect: () => startNew() });
    openPopupMenu({ items, x: rect.left, y: rect.bottom + 4, logName: LOG_NAME });
  };

  const openActions = (event) => {
    const rect = (event?.currentTarget ?? more).getBoundingClientRect();
    const name = current();
    openPopupMenu({
      items: [
        { label: "Rename...", disabled: !name, onSelect: () => beginRename() },
        { label: "Delete...", danger: true, disabled: !name, onSelect: () => confirmDelete() },
      ],
      x: rect.right - 200,
      y: rect.bottom + 4,
      logName: LOG_NAME,
    });
  };

  const endRename = () => {
    renameBox.style.display = "none";
    picker.style.display = "";
  };

  const beginRename = () => {
    const name = current();
    if (!name) return;
    say("");
    renameBox.value = name;
    picker.style.display = "none";
    renameBox.style.display = "";
    renameBox.focus();
    renameBox.select();
  };

  const commitRename = async () => {
    const from = current();
    const to = renameBox.value.trim();
    endRename();
    if (!from || !to || to === from) return;
    if (!usableName(to)) {
      say(`"${to}" cannot be used. Use letters, digits, spaces, dots, dashes or underscores.`, LOOK.danger);
      return;
    }
    const saved = (await savedRows()).some((row) => String(row.name) === from);
    if (saved) {
      const answer = await ask(`action=rename&name=${encodeURIComponent(from)}&to=${encodeURIComponent(to)}`, "POST");
      if (answer.status !== 200) {
        say(String(answer.body?.error ?? "The conversation could not be renamed."), LOOK.danger);
        return;
      }
    }
    fresh = false;
    lastSession = to;
    setWidget(node, conversationWidget(), to);
    drawnKey = "";
    showHeader();
  };

  renameBox.addEventListener("keydown", (event) => {
    event.stopPropagation();
    if (event.key === "Enter") {
      event.preventDefault();
      commitRename();
    } else if (event.key === "Escape") {
      event.preventDefault();
      endRename();
    }
  });
  renameBox.addEventListener("blur", () => {
    if (renameBox.style.display !== "none") endRename();
  });

  const confirmDelete = () => {
    const name = current();
    if (!name) return;
    notice.textContent = "";
    notice.appendChild(el("span", "flex:1 1 auto;min-width:0;overflow-wrap:anywhere", `Delete "${name}" and its saved messages?`));
    const yes = button("Delete", async () => {
      const answer = await ask(`action=delete&name=${encodeURIComponent(name)}`, "POST");
      if (answer.status !== 200) {
        say(String(answer.body?.error ?? "The conversation could not be deleted."), LOOK.danger);
        return;
      }
      say("");
      if (lastSession === name) lastSession = "";
      if (current() === name || chosen() === name) startNew();
    }, { kind: "danger", logName: LOG_NAME });
    const no = button("Cancel", () => say(""), { kind: "tool", logName: LOG_NAME });
    notice.append(yes, no);
    notice.style.display = "flex";
  };

  const submit = async () => {
    const text = box.value.trim();
    const fromLink = wired(node, messageName);
    if (!text && !fromLink) return;
    say("");
    // A message with no conversation named starts one, named from its words.
    if (!current() && !wired(node, conversationName) && (fresh || memorySession(node, memoryName) !== null)) {
      const taken = new Set((await savedRows()).map((row) => String(row.name).toLowerCase()));
      if (disposed) return;
      setWidget(node, conversationWidget(), nameFrom(text, taken));
    }
    fresh = false;
    showHeader();
    if (!fromLink) setWidget(node, widgetNamed(node, messageName), text);
    if (empty.parentElement) empty.remove();
    if (text) transcript.appendChild(bubble({ role: "user", text, pictures: 0 }).row);
    const reply = bubble({ role: "assistant", text: "...", pictures: 0 });
    reply.body.style.color = LOOK.muted;
    transcript.appendChild(reply.row);
    pending = reply;
    sentText = text;
    scrollToEnd();
    box.value = "";
    setDisabled(send, true);
    Promise.resolve().then(() => app.queuePrompt(0)).catch((error) => {
      console.error(`[${LOG_NAME}] The graph could not be queued:`, error);
      setDisabled(send, false);
    });
  };

  // Enter sends, Shift+Enter starts a new line; keys typed here never reach the graph.
  box.addEventListener("keydown", (event) => {
    event.stopPropagation();
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submit();
    }
  });
  for (const target of [box, renameBox]) {
    for (const kind of ["keyup", "keypress", "paste", "copy", "cut"]) {
      target.addEventListener(kind, (event) => event.stopPropagation());
    }
  }

  // The reply as it is written, from the text the node streams while it runs.
  const onProgressText = (event) => {
    if (disposed || !pending) return;
    const detail = event?.detail ?? {};
    if (String(detail.nodeId ?? "") !== executionId(node)) return;
    pending.body.textContent = String(detail.text ?? "");
    pending.body.style.color = LOOK.text;
    scrollToEnd();
  };
  api.addEventListener("progress_text", onProgressText);

  const stopFinished = onNodeFinished(node, () => {
    setDisabled(send, false);
    sentText = "";
    refresh();
  });
  const stopEnded = onRunEnded(() => {
    setDisabled(send, false);
    if (!pending) return;
    // A run that ended without answering gives the message back to the box.
    if (sentText && !box.value) box.value = sentText;
    sentText = "";
    pending = null;
    drawnKey = "";
    refresh();
  });
  const stopSettling = whenReadable(node, () => refresh());

  const releaseWheel = captureWheel(root, (event) => {
    const target = event.target instanceof Element ? event.target : root;
    const region = target.closest?.("textarea") ?? transcript;
    const before = region.scrollTop;
    region.scrollTop += wheelPixels(event, region).y;
    return region.scrollTop !== before;
  });

  for (const name of [messageName, conversationName]) {
    const widget = widgetNamed(node, name);
    if (widget) setWidgetHidden(widget, true);
  }
  showHeader();
  showWiring();

  return {
    element: root,
    height,
    minWidth: MIN_WIDTH,
    update: () => refresh(),
    clear: () => draw([]),
    refresh: () => refresh(),
    dispose() {
      if (disposed) return;
      disposed = true;
      api.removeEventListener?.("progress_text", onProgressText);
      if (typeof stopFinished === "function") stopFinished();
      if (typeof stopEnded === "function") stopEnded();
      stopSettling();
      releaseWheel();
    },
  };
}
