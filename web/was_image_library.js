/**
 * The picture library drawn on Image Library Search: the folder tree with each folder's index
 * state, a bar for the search settings, and the matches for the node's prompt, asked of the
 * server as the prompt changes.
 */

import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { findWidget, writeWidget } from "./interface/audio_view.js";
import { CARD_LOOK } from "./interface/card.js";
import { fetchWithin } from "./interface/request.js";
import { fetchRunResult } from "./interface/run_result.js";
import { themeVar } from "./interface/theme.js";
import { setWidgetHidden } from "./interface/visibility.js";
import { appendInterfaceWidget, chainWidgetCallback } from "./interface/widget.js";
import { LOOK, SANS } from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.ImageLibrary";
const NODE_ID = "WASImageLibrarySearch";

const UI_WIDGET_NAME = "was_image_library_ui";
const UI_WIDGET_TYPE = "was_image_library";

const SEARCH_ROUTE = "/was/interface/api/image_library";
const PAUSE_ROUTE = "/was/interface/api/pause";
const THUMB_ROUTE = "/was/interface/api/file_thumbnail";
const THUMB_EDGE = 192;

// Height and width in node units the panel opens at.
const PANEL_HEIGHT = 470;
const PANEL_WIDTH = 380;

// Milliseconds the prompt and the bar settle before the server is asked again.
const SETTLE = 450;

// Narrowest a match tile is drawn, in CSS pixels.
const TILE_MIN = 96;

// The widgets the bar sets, folded off the canvas while the panel draws them.
const BAR_WIDGETS = ["view", "filter", "top_k", "one_per_group", "pause"];

// The widgets the search reads, in the names the server takes them under.
const SETTINGS = [
  "folder", "folders", "captions", "recursive", "max_files", "text_model", "picture_model",
  "prompt", "view", "filter", "top_k", "one_per_group", "picture_weight", "dimensions",
];

// Index states as the server names them, and the colour each chip is drawn in.
const STATE_LOOK = {
  indexed: () => LOOK.success,
  "partly indexed": () => LOOK.warning,
  "not indexed": () => themeVar("fgMuted"),
};

const VIEW_NAMES = {
  auto: "View: from prompt",
  any: "View: any",
  front: "Front",
  side: "Side",
  diag: "Three-quarter",
  high: "High angle",
  low: "Low angle",
  back: "Back",
};

/**
 * An element with a style and a text.
 *
 * @param {string} tag - The tag.
 * @param {string} css - Its inline style.
 * @param {string} [text] - Its text.
 * @returns {HTMLElement} The element.
 */
function el(tag, css, text = "") {
  const element = document.createElement(tag);
  element.style.cssText = css;
  if (text) element.textContent = text;
  return element;
}

/**
 * Keep a control's pointer and key presses from reaching the graph canvas.
 *
 * @param {HTMLElement} element - The control.
 * @returns {HTMLElement} The same control.
 */
function claim(element) {
  for (const type of ["pointerdown", "wheel", "keydown"]) {
    element.addEventListener(type, (event) => event.stopPropagation());
  }
  return element;
}

/**
 * The tab's client id, which a held run is resumed under.
 *
 * @returns {string} The id.
 */
function clientId() {
  return String(api?.clientId || api?.initialClientId || "");
}

/**
 * Whether a widget's input is fed by a link, so its value is not on the node.
 *
 * @param {object} node - The node.
 * @param {string} name - The input's name.
 * @returns {boolean} True when linked.
 */
function linked(node, name) {
  const slot = (node.inputs ?? []).find((input) => input?.name === name || input?.widget?.name === name);
  return slot?.link != null;
}

/**
 * A button styled for the bar.
 *
 * @param {string} text - Its label.
 * @param {string} hint - What it does, shown on hover.
 * @returns {HTMLButtonElement} The button.
 */
function barButton(text, hint) {
  const button = claim(el(
    "button",
    `flex:0 0 auto;height:24px;padding:0 8px;border-radius:5px;cursor:pointer;font:inherit;`
      + `border:1px solid ${themeVar("border")};background:${CARD_LOOK.inset};color:${CARD_LOOK.text}`,
    text,
  ));
  button.type = "button";
  button.title = hint;
  return button;
}

/**
 * Build the library panel.
 *
 * @param {object} node - The node the panel belongs to.
 * @returns {{element: HTMLElement, height: number, minWidth: number, refresh: () => void,
 *   held: (on: boolean) => void, dispose: () => void}} The panel.
 */
function createLibraryPanel(node) {
  const root = el(
    "div",
    "box-sizing:border-box;width:100%;height:100%;display:flex;flex-direction:column;gap:6px;"
      + `padding:6px;font:11px ${SANS};color:${CARD_LOOK.text};overflow:hidden`,
  );

  // The folder tree.
  const treeHead = el("div", "flex:0 0 auto;display:flex;align-items:center;gap:6px");
  const treeToggle = barButton("▾", "Show or hide folders");
  treeToggle.style.width = "24px";
  treeToggle.style.padding = "0";
  const treeTitle = el("div", "flex:1 1 auto;min-width:0;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis");
  const allButton = barButton("All", "Search every folder");
  const refreshButton = barButton("↻", "Read folders again");
  treeHead.append(treeToggle, treeTitle, allButton, refreshButton);
  const tree = claim(el(
    "div",
    "flex:0 1 auto;max-height:132px;min-height:0;overflow:auto;display:flex;flex-direction:column;"
      + `gap:1px;border:1px solid ${themeVar("border")};border-radius:6px;padding:3px;`
      + `background:${CARD_LOOK.inset};scrollbar-width:thin`,
  ));

  // The bar of search settings.
  const bar = el("div", "flex:0 0 auto;display:flex;flex-wrap:wrap;gap:4px;align-items:center");
  const viewSelect = claim(el(
    "select",
    `flex:0 0 auto;height:24px;border-radius:5px;font:inherit;border:1px solid ${themeVar("border")};`
      + `background:${CARD_LOOK.inset};color:${CARD_LOOK.text}`,
  ));
  viewSelect.title = "Camera view sent on";
  for (const [value, text] of Object.entries(VIEW_NAMES)) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = text;
    viewSelect.appendChild(option);
  }
  const filterInput = claim(el(
    "input",
    `flex:1 1 90px;min-width:70px;height:22px;padding:0 6px;border-radius:5px;font:inherit;`
      + `border:1px solid ${themeVar("border")};background:${CARD_LOOK.inset};color:${CARD_LOOK.text}`,
  ));
  filterInput.placeholder = "filter: front -couples";
  filterInput.title = "Words to keep or -drop";
  const less = barButton("−", "Fewer matches");
  const count = el("span", "flex:0 0 auto;min-width:22px;text-align:center;font-variant-numeric:tabular-nums");
  const more = barButton("+", "More matches");
  const groupButton = barButton("One per pose", "One angle per pose");
  const pauseButton = barButton("Pause", "Hold the run to pick");
  bar.append(viewSelect, filterInput, less, count, more, groupButton, pauseButton);

  // The matches.
  const grid = claim(el(
    "div",
    "flex:1 1 auto;min-height:0;overflow:auto;display:grid;gap:5px;align-content:start;"
      + `grid-template-columns:repeat(auto-fill,minmax(${TILE_MIN}px,1fr));scrollbar-width:thin`,
  ));
  const footer = el("div", "flex:0 0 auto;display:flex;align-items:center;gap:6px;min-height:26px");
  const status = el("div", `flex:1 1 auto;min-width:0;color:${themeVar("fgMuted")};white-space:nowrap;overflow:hidden;text-overflow:ellipsis`);
  const pinChip = barButton("", "Unpin, send the closest");
  pinChip.style.display = "none";
  const continueButton = barButton("Continue", "Send the pick on");
  continueButton.style.display = "none";
  continueButton.style.background = LOOK.accent;
  continueButton.style.color = themeVar("panelBg");
  continueButton.style.fontWeight = "650";
  footer.append(status, pinChip, continueButton);

  root.append(treeHead, tree, bar, grid, footer);

  let disposed = false;
  let serial = 0;
  let timer = 0;
  let isHeld = false;
  let picked = "";
  let folders = [];
  let matches = [];
  let treeOpen = true;

  const value = (name) => findWidget(node, name)?.value;
  const write = (name, next) => {
    writeWidget(node, name, next);
    schedule(0);
  };

  /**
   * Draw the bar from the widgets it sets.
   *
   * @returns {void}
   */
  const drawBar = () => {
    viewSelect.value = String(value("view") ?? "auto");
    if (document.activeElement !== filterInput) filterInput.value = String(value("filter") ?? "");
    count.textContent = String(value("top_k") ?? 8);
    const on = (button, active) => {
      button.style.background = active ? `color-mix(in srgb, ${LOOK.accent} 30%, ${CARD_LOOK.inset})` : CARD_LOOK.inset;
      button.style.borderColor = active ? LOOK.accent : themeVar("border");
    };
    on(groupButton, value("one_per_group") !== false);
    on(pauseButton, value("pause") === true);
  };

  /**
   * The folders the node searches, as the labels its `folders` widget names.
   *
   * @returns {Set<string>|null} The labels, or null for every folder.
   */
  const ticked = () => {
    const text = String(value("folders") ?? "").trim();
    if (!text) return null;
    return new Set(text.split("\n").map((line) => line.trim()).filter(Boolean));
  };

  /**
   * Draw the folder tree from the server's answer.
   *
   * @returns {void}
   */
  const drawTree = () => {
    tree.textContent = "";
    tree.style.display = treeOpen ? "flex" : "none";
    treeToggle.textContent = treeOpen ? "▾" : "▸";
    const chosen = ticked();
    const searched = folders.filter((folder) => !chosen || chosen.has(folder.label));
    const pictures = searched.reduce((sum, folder) => sum + folder.count, 0);
    treeTitle.textContent = folders.length
      ? `${searched.length} of ${folders.length} folder${folders.length === 1 ? "" : "s"} · ${pictures.toLocaleString()} pictures`
      : "No folders read yet";
    for (const folder of folders) {
      const depth = folder.label === "." ? 0 : folder.label.split("/").length - 1;
      const row = el(
        "label",
        `display:flex;align-items:center;gap:6px;padding:2px 4px 2px ${4 + depth * 14}px;border-radius:4px;cursor:pointer`,
      );
      row.title = `${folder.label === "." ? "This folder's own pictures" : folder.label}\n${folder.source}`;
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = !chosen || chosen.has(folder.label);
      box.addEventListener("change", () => {
        const next = new Set(chosen ?? folders.map((entry) => entry.label));
        if (box.checked) next.add(folder.label);
        else next.delete(folder.label);
        const all = folders.every((entry) => next.has(entry.label));
        write("folders", all ? "" : [...next].join("\n"));
        drawTree();
      });
      const name = el("span", "flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis",
        folder.label === "." ? "(this folder)" : folder.label.split("/").pop());
      const amount = el("span", `flex:0 0 auto;color:${themeVar("fgMuted")};font-variant-numeric:tabular-nums`,
        folder.count.toLocaleString());
      const chip = el(
        "span",
        `flex:0 0 auto;padding:0 6px;border-radius:9px;font-size:10px;border:1px solid ${(STATE_LOOK[folder.state] ?? STATE_LOOK["not indexed"])()};`
          + `color:${(STATE_LOOK[folder.state] ?? STATE_LOOK["not indexed"])()}`,
        folder.state,
      );
      chip.title = folder.state === "indexed" ? "Searched from its index"
        : "Run the node to index";
      row.append(box, name, amount, chip);
      tree.appendChild(row);
    }
  };

  /**
   * Draw the matches.
   *
   * @returns {void}
   */
  const drawGrid = () => {
    const scrolled = grid.scrollTop;
    grid.textContent = "";
    const pinned = String(value("pinned") ?? "");
    for (const [index, match] of matches.entries()) {
      const chosenNow = isHeld ? match.key === picked : match.key === pinned;
      const tile = claim(el(
        "button",
        "display:flex;flex-direction:column;gap:2px;min-width:0;padding:3px;margin:0;cursor:pointer;"
          + `border-radius:6px;font:inherit;text-align:left;background:${CARD_LOOK.inset};color:${CARD_LOOK.text};`
          + `border:1px solid ${chosenNow ? LOOK.accent : themeVar("border")};`
          + `box-shadow:${chosenNow ? `0 0 0 1px ${LOOK.accent}` : "none"}`,
      ));
      tile.type = "button";
      tile.title = `${match.name}\n${match.caption}\n\n${isHeld ? "Click to pick" : (chosenNow ? "Click to unpin" : "Click to pin")}`;
      const frame = el("div", `width:100%;aspect-ratio:3/4;border-radius:4px;overflow:hidden;background:${themeVar("panelBg")}`);
      if (match.label) {
        const image = document.createElement("img");
        image.alt = "";
        image.loading = "lazy";
        image.decoding = "async";
        image.draggable = false;
        image.style.cssText = "display:block;width:100%;height:100%;object-fit:contain;pointer-events:none";
        image.addEventListener("error", () => image.remove(), { once: true });
        image.src = api.apiURL(`${THUMB_ROUTE}?label=${encodeURIComponent(match.label)}&edge=${THUMB_EDGE}`);
        frame.appendChild(image);
      }
      const head = el("div", "display:flex;justify-content:space-between;gap:4px;font-size:10px");
      head.append(
        el("span", "font-weight:650", `${chosenNow ? "📌 " : ""}#${index + 1}`),
        el("span", `color:${themeVar("fgMuted")}`, Number(match.score).toFixed(2)),
      );
      const name = el("div", `font-size:10px;color:${themeVar("fgMuted")};white-space:nowrap;overflow:hidden;text-overflow:ellipsis`, match.name);
      tile.append(frame, head, name);
      tile.addEventListener("click", (event) => {
        event.stopPropagation();
        if (isHeld) {
          picked = match.key;
          drawGrid();
          drawFooter();
          return;
        }
        writeWidget(node, "pinned", match.key === pinned ? "" : match.key);
        drawGrid();
        drawFooter();
      });
      grid.appendChild(tile);
    }
    grid.scrollTop = scrolled;
  };

  let note = "";
  let searched = 0;

  /**
   * Draw the status line, the pinned picture and Continue.
   *
   * @returns {void}
   */
  const drawFooter = () => {
    const pinned = String(value("pinned") ?? "");
    if (isHeld) {
      const choice = matches.find((match) => match.key === picked);
      status.textContent = choice ? `Picked ${choice.name}` : "Paused: pick a match";
      status.style.color = LOOK.accent;
    } else {
      status.textContent = note || (matches.length
        ? `${matches.length} matches of ${searched.toLocaleString()} searched`
        : "No matches yet");
      status.style.color = themeVar("fgMuted");
    }
    pinChip.style.display = pinned && !isHeld ? "inline-block" : "none";
    pinChip.textContent = pinned ? `📌 ${pinned.split("/").pop().replace(/\.[^.]+$/, "")} ✕` : "";
    continueButton.style.display = isHeld ? "inline-block" : "none";
    continueButton.disabled = isHeld && !picked && !matches.length;
  };

  /**
   * The last run's matches, for a prompt that arrives through a link.
   *
   * @returns {Promise<void>}
   */
  const fromLastRun = async () => {
    const answer = await fetchRunResult(node);
    const body = (answer?.result?.bodies ?? []).find((entry) => entry?.name === "matches");
    matches = String(body?.text ?? "").split("\n").filter(Boolean).map((line) => {
      const [, label = "", score = "", name = "", caption = "", key = ""] = line.split("\t");
      return { label, score: Number.parseFloat(score) || 0, name, caption, key };
    });
    searched = matches.length;
    note = matches.length ? "Linked prompt: last run's matches" : "Linked prompt: run to search";
  };

  /**
   * Ask the server for the folders and the matches the node's settings name.
   *
   * @returns {Promise<void>}
   */
  const search = async () => {
    const asked = ++serial;
    const body = {};
    for (const name of SETTINGS) body[name] = value(name);
    try {
      if (linked(node, "prompt")) {
        body.prompt = "";
      }
      const response = await fetchWithin(SEARCH_ROUTE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const reply = await response.json();
      if (disposed || asked !== serial) return;
      if (!response.ok) throw new Error(reply?.error ?? response.status);
      folders = Array.isArray(reply.folders) ? reply.folders : [];
      if (linked(node, "prompt")) {
        await fromLastRun();
      } else {
        matches = Array.isArray(reply.matches) ? reply.matches : [];
        searched = Number(reply.searched) || 0;
        note = reply.note ? reply.note.charAt(0).toUpperCase() + reply.note.slice(1) : "";
      }
      if (isHeld && !matches.some((match) => match.key === picked)) picked = matches[0]?.key ?? "";
    } catch (error) {
      if (disposed || asked !== serial) return;
      note = "Search unreachable: run the node";
      console.error(`[${EXT_NAME}] Failed to search the library:`, error);
    }
    drawBar();
    drawTree();
    drawGrid();
    drawFooter();
    node.setDirtyCanvas?.(true, false);
  };

  /**
   * Search once the settings have stopped changing.
   *
   * @param {number} [delay] - Milliseconds to wait.
   * @returns {void}
   */
  const schedule = (delay = SETTLE) => {
    clearTimeout(timer);
    timer = setTimeout(() => search(), delay);
  };

  treeToggle.addEventListener("click", () => {
    treeOpen = !treeOpen;
    drawTree();
  });
  allButton.addEventListener("click", () => {
    write("folders", "");
    drawTree();
  });
  refreshButton.addEventListener("click", () => schedule(0));
  viewSelect.addEventListener("change", () => write("view", viewSelect.value));
  filterInput.addEventListener("input", () => {
    writeWidget(node, "filter", filterInput.value);
    schedule();
  });
  less.addEventListener("click", () => write("top_k", Math.max(1, Number(value("top_k") ?? 8) - 1)));
  more.addEventListener("click", () => write("top_k", Math.min(64, Number(value("top_k") ?? 8) + 1)));
  groupButton.addEventListener("click", () => write("one_per_group", value("one_per_group") === false));
  pauseButton.addEventListener("click", () => {
    writeWidget(node, "pause", value("pause") !== true);
    drawBar();
  });
  pinChip.addEventListener("click", () => {
    writeWidget(node, "pinned", "");
    drawGrid();
    drawFooter();
  });
  continueButton.addEventListener("click", async () => {
    const key = picked || matches[0]?.key || "";
    try {
      const answer = await fetchWithin(PAUSE_ROUTE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: String(node.id), action: "resume", value: key, client_id: clientId() }),
      });
      if (!answer.ok) console.error(`[${EXT_NAME}] The run was not resumed:`, answer.status);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to resume the run:`, error);
    }
  });

  // The prompt is read as it is typed, and any other setting the search reads as it changes.
  for (const name of SETTINGS) chainWidgetCallback(node, name, () => schedule(), EXT_NAME);
  const promptBox = findWidget(node, "prompt");
  promptBox?.element?.addEventListener?.("input", () => schedule());
  promptBox?.inputEl?.addEventListener?.("input", () => schedule());

  drawBar();
  drawTree();
  drawFooter();
  schedule(0);

  return {
    element: root,
    height: PANEL_HEIGHT,
    minWidth: PANEL_WIDTH,
    refresh: () => schedule(0),
    held(on) {
      isHeld = on;
      if (on && !matches.some((match) => match.key === picked)) {
        picked = String(value("pinned") ?? "") || matches[0]?.key || "";
      }
      if (!on) picked = "";
      schedule(0);
      drawGrid();
      drawFooter();
    },
    dispose() {
      disposed = true;
      clearTimeout(timer);
      serial += 1;
    },
  };
}

const panels = new Map();

/**
 * Whether a node is holding its run, asked of the server for a node added after the hold began.
 *
 * @param {object} node - The node.
 * @returns {Promise<boolean>} True when it is waiting.
 */
async function isWaiting(node) {
  try {
    const answer = await fetchWithin(`${PAUSE_ROUTE}?client_id=${encodeURIComponent(clientId())}`);
    const body = await answer.json();
    return (body?.waiting ?? []).some((hold) => String(hold.node_id) === String(node.id));
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read whether the run is held:`, error);
    return false;
  }
}

app.registerExtension({
  name: EXT_NAME,

  async setup() {
    api.addEventListener("was-pause", ({ detail }) => panels.get(String(detail?.node_id))?.held(true));
    api.addEventListener("was-pause-done", ({ detail }) => panels.get(String(detail?.node_id))?.held(false));
    // Every panel asks again once a run has finished.
    api.addEventListener("execution_success", () => {
      for (const panel of panels.values()) panel.refresh();
    });
  },

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_image_library_wrapped) return;
    proto.__was_image_library_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        for (const name of BAR_WIDGETS) {
          const widget = findWidget(this, name);
          if (widget) setWidgetHidden(widget, true);
        }
        const panel = createLibraryPanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnAdded = this.onAdded;
        this.onAdded = function (...args) {
          const added = originalOnAdded?.apply(this, args);
          panels.set(String(this.id), panel);
          isWaiting(this).then((waiting) => { if (waiting) panel.held(true); });
          return added;
        };
        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          panels.delete(String(this.id));
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the panel:`, error);
      }
      return result;
    };
  },
});
