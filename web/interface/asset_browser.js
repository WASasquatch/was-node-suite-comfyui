/**
 * A browser of the pictures, clips and sounds the pack may read, for picking one for a role.
 *
 * A pick arrives on `onChoose` with the role's id; a dragged tile carries its label under
 * `DRAG_TYPE`. Sizes are CSS pixels.
 */

import { api } from "../../../scripts/api.js";
import { elementPoint } from "./pointer.js";
import { readableBytes } from "./report_panel.js";
import { UPLOAD_TIMEOUT, fetchWithin } from "./request.js";
import {
  FIELD, FLOATING_PANEL, LABEL, LOOK, SANS, SCROLLBARS, button, focusRing,
} from "./window_kit.js";

const LOG_NAME = "WASNodeSuite.AssetBrowser";

// The routes the browser reads, and the one it uploads through.
const LISTING_ROUTE = "/was/interface/api/file_listing";
const THUMBNAIL_ROUTE = "/was/interface/api/file_thumbnail";
const UPLOAD_ROUTE = "/upload/image";

// Entries one listing asks for.
const LISTING_LIMIT = 5000;

// Tiles drawn at most.
const TILE_CAP = 600;

// The thumbnail box a caller gets when it names none, and the edge the picture is asked for at.
const THUMB_WIDTH = 120;
const THUMB_HEIGHT = 90;
const THUMB_EDGE = 192;

/** The drag data type a tile's label is carried under. */
export const DRAG_TYPE = "application/x-was-file";

// Space between tiles, and around the grid.
const TILE_GAP = 8;

// The glyph drawn where there is no picture, in pixels a side.
const GLYPH_SIZE = 28;

// How long typing settles before the grid is filtered, in milliseconds.
const SEARCH_SETTLE_MS = 120;

// Pixels between a tile and the menu beside it.
const MENU_GAP = 4;

/** The suffixes each kind covers. */
export const KIND_EXTENSIONS = Object.freeze({
  picture: Object.freeze([".jpeg", ".jpg", ".png", ".tiff", ".gif", ".bmp", ".webp"]),
  clip: Object.freeze([
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv",
  ]),
  sound: Object.freeze([".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".wma"]),
});

// The kinds in the order the chips are drawn, and what each chip says.
const KIND_ORDER = Object.freeze(["picture", "clip", "sound"]);
const KIND_LABELS = Object.freeze({ picture: "Pictures", clip: "Clips", sound: "Sounds" });

const SVG_NS = "http://www.w3.org/2000/svg";

// The glyphs drawn where a picture is not: a film strip, a waveform and a broken picture,
// as paths in a 24 unit box.
const GLYPH_PATHS = Object.freeze({
  clip: Object.freeze(["M3 5h18v14H3z", "M3 9h18", "M3 15h18", "M8 5v14", "M16 5v14"]),
  sound: Object.freeze(["M4 10v4", "M8 7v10", "M12 4v16", "M16 8v8", "M20 10v4"]),
  broken: Object.freeze(["M4 5h16v14H4z", "M4 17l5-5 3 3 3-3 5 5", "M13 5l-2 4 3 2-2 4"]),
});

/**
 * The suffix of one listing entry, lowercased with its dot.
 *
 * @param {object} entry - A row of the listing.
 * @returns {string} The suffix, or an empty string for a name with none.
 */
function suffixOf(entry) {
  const name = String(entry?.relative || "")
    || String(entry?.label ?? "").replace(/\s\[[^\]]*\]$/, "");
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot).toLowerCase();
}

/**
 * Which kind one listing entry is.
 *
 * @param {object} entry - A row of the listing.
 * @returns {string|null} `picture`, `clip` or `sound`, or null for a suffix of none of them.
 */
export function kindOf(entry) {
  const suffix = suffixOf(entry);
  for (const kind of KIND_ORDER) {
    if (KIND_EXTENSIONS[kind].includes(suffix)) return kind;
  }
  return null;
}

/**
 * The file name of one entry, without its folders.
 *
 * @param {object} entry - A row of the listing.
 * @returns {string} The name.
 */
function baseName(entry) {
  const path = String(entry?.relative || "")
    || String(entry?.label ?? "").replace(/\s\[[^\]]*\]$/, "");
  return path.split(/[\\/]/).pop() || path;
}

/**
 * A modification time as a date and a time.
 *
 * @param {number} mtime - Seconds since the epoch.
 * @returns {string} `YYYY-MM-DD HH:MM`, or an empty string for no time.
 */
function dateOf(mtime) {
  const stamp = Number(mtime);
  if (!Number.isFinite(stamp) || stamp <= 0) return "";
  const date = new Date(stamp * 1000);
  const pad = (value) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
    + ` ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/**
 * The thumbnail route for one entry, stamped with its modification time.
 *
 * @param {object} entry - A row of the listing.
 * @returns {string} The URL an image element loads.
 */
function thumbnailUrl(entry) {
  const stamp = encodeURIComponent(String(entry.mtime ?? ""));
  return api.apiURL(
    `${THUMBNAIL_ROUTE}?label=${encodeURIComponent(entry.label)}&edge=${THUMB_EDGE}&stamp=${stamp}`,
  );
}

/**
 * One glyph for a tile with no picture.
 *
 * @param {string} name - `clip`, `sound` or `broken`.
 * @returns {SVGElement} The glyph, drawn in the current text colour.
 */
function createGlyph(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(GLYPH_SIZE));
  svg.setAttribute("height", String(GLYPH_SIZE));
  svg.setAttribute("aria-hidden", "true");
  svg.style.cssText = "display:block;fill:none;stroke:currentColor;stroke-width:1.5;"
    + "stroke-linecap:round;stroke-linejoin:round;pointer-events:none";
  for (const path of GLYPH_PATHS[name] ?? GLYPH_PATHS.broken) {
    const shape = document.createElementNS(SVG_NS, "path");
    shape.setAttribute("d", path);
    svg.appendChild(shape);
  }
  return svg;
}

/**
 * The roles a caller passed, each with a string id, a label and the kinds it takes.
 *
 * @param {unknown} list - What the caller passed.
 * @returns {Array<{id: string, label: string, kinds: string[]|null}>} The roles with an id,
 *   in the order given. `kinds` is null for a role that takes any kind.
 */
function normaliseRoles(list) {
  if (!Array.isArray(list)) return [];
  const roles = [];
  for (const role of list) {
    const id = String(role?.id ?? "");
    if (!id) continue;
    const kinds = Array.isArray(role?.kinds) ? role.kinds.map(String) : null;
    roles.push({ id, label: String(role?.label ?? id), kinds });
  }
  return roles;
}

/**
 * The kinds a caller asked for, in drawing order.
 *
 * @param {unknown} list - What the caller passed.
 * @returns {string[]} The known kinds named, or all three when none is.
 */
function normaliseKinds(list) {
  if (!Array.isArray(list)) return [...KIND_ORDER];
  const kinds = KIND_ORDER.filter((kind) => list.includes(kind));
  return kinds.length ? kinds : [...KIND_ORDER];
}

/**
 * One text field on the toolbar.
 *
 * @param {HTMLElement} field - An input or a select.
 * @param {string} flex - Its share of the toolbar row.
 * @returns {void}
 */
function styleField(field, flex) {
  field.style.cssText = `${FIELD};flex:${flex};font-size:12px;padding:6px 9px`;
  focusRing(field);
}

/**
 * Whether a drag carries files from outside the page.
 *
 * @param {DragEvent} event - The drag event.
 * @returns {boolean} True when the drag holds files.
 */
export function carriesFiles(event) {
  return Array.from(event?.dataTransfer?.types ?? []).includes("Files");
}

/**
 * The label the file listing names an uploaded file by.
 *
 * @param {object} answer - What the upload route answered, `{name, subfolder, type}`.
 * @returns {string} The label, as `shots/one.png [input]`.
 */
export function uploadedLabel(answer) {
  const folder = String(answer?.subfolder ?? "");
  const name = String(answer?.name ?? "");
  return `${folder ? `${folder}/` : ""}${name} [${String(answer?.type || "input")}]`;
}

/**
 * Draw a kind chip on or off.
 *
 * @param {HTMLButtonElement} chip - The chip.
 * @param {boolean} on - Whether its kind is listed.
 * @returns {void}
 */
function paintChip(chip, on) {
  chip.setAttribute("aria-pressed", on ? "true" : "false");
  chip.style.background = on ? LOOK.hover : "transparent";
  chip.style.borderColor = LOOK.border;
  chip.style.color = on ? LOOK.text : LOOK.muted;
}

/**
 * A pill that switches one kind of file in or out of the grid.
 *
 * @param {string} label - What it says.
 * @param {string} hint - What the hover says.
 * @param {() => void} onPress - What a press does.
 * @returns {HTMLButtonElement} The pill.
 */
function createKindPill(label, hint, onPress) {
  const pill = document.createElement("button");
  pill.type = "button";
  pill.title = hint;
  pill.textContent = label;
  pill.style.cssText = `flex:0 0 auto;padding:2px 10px;border-radius:999px;cursor:pointer;`
    + `font:12px/18px ${SANS};border:1px solid ${LOOK.border}`;
  pill.addEventListener("click", (event) => {
    event.stopPropagation();
    onPress();
  });
  return pill;
}

/**
 * Build a browser of the files the pack may read.
 *
 * @param {object} [options] - How the browser is built.
 * @param {(entry: object, role: string) => void} [options.onChoose] - Called with the entry
 *   picked, `{label, relative, tag, size, mtime, kind}`, and the id of the role it was
 *   picked for.
 * @param {Array<{id: string, label: string}>} [options.roles] - What the "Use as" menu offers.
 * @param {string[]} [options.kinds] - Which of `picture`, `clip` and `sound` are listed. All
 *   three by default.
 * @param {string} [options.logName] - The prefix a failure is logged under.
 * @param {number} [options.thumbWidth] - The narrowest a tile's picture is drawn, 120 by default.
 * @param {number} [options.thumbHeight] - The height of a tile's picture, 90 by default.
 * @param {(entry: object) => void} [options.onSelect] - Called with the entry of a tile pressed.
 * @param {(entry: object) => void} [options.onDragStart] - Called as a tile is dragged out.
 * @param {() => void} [options.onDragEnd] - Called when that drag ends, dropped or not.
 * @param {boolean} [options.acceptFiles] - Whether files dropped on the grid are uploaded.
 * @returns {{element: HTMLElement, refresh: () => Promise<void>,
 *   setRoles: (roles: Array<{id: string, label: string}>) => void,
 *   setKinds: (kinds: string[]) => void, upload: (files: File[]) => Promise<object[]>,
 *   dispose: () => void}} The browser: its element, the call that reads the listing again, the
 *   two that re-shape it, the one that uploads, and the one that empties it.
 */
export function createAssetBrowser(options = {}) {
  const logName = options.logName || LOG_NAME;
  const onChoose = typeof options.onChoose === "function" ? options.onChoose : null;
  const onSelect = typeof options.onSelect === "function" ? options.onSelect : null;
  const onDragStart = typeof options.onDragStart === "function" ? options.onDragStart : null;
  const onDragEnd = typeof options.onDragEnd === "function" ? options.onDragEnd : null;
  const acceptFiles = options.acceptFiles === true;
  const thumbWidth = Number(options.thumbWidth) > 0 ? Number(options.thumbWidth) : THUMB_WIDTH;
  const thumbHeight = Number(options.thumbHeight) > 0 ? Number(options.thumbHeight) : THUMB_HEIGHT;

  let roles = normaliseRoles(options.roles);
  let kinds = normaliseKinds(options.kinds);
  let enabled = new Set(kinds);
  let entries = [];
  let roots = [];
  let truncated = false;
  let shown = [];
  let passing = 0;
  let tiles = [];
  let pluses = [];
  let selected = -1;
  let menuFor = -1;
  let query = "";
  let rootTag = "";
  let controller = null;
  let searchTimer = 0;
  let uploading = false;
  let disposed = false;

  const root = document.createElement("div");
  root.style.cssText = [
    "position:relative",
    "width:100%",
    "height:100%",
    "display:flex",
    "flex-direction:column",
    "min-height:0",
    "box-sizing:border-box",
    `color:${LOOK.text}`,
    `font:12px/1.4 ${SANS}`,
  ].join(";");

  const toolbar = document.createElement("div");
  toolbar.style.cssText = "flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;"
    + `gap:6px;padding:8px 12px 10px;border-bottom:1px solid ${LOOK.border}`;

  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "search by name";
  search.setAttribute("aria-label", "Search files by name");
  styleField(search, "1 1 140px");
  search.style.minWidth = "100px";

  const chipRow = document.createElement("div");
  chipRow.style.cssText = "flex:0 0 auto;display:flex;gap:4px";

  const rootMenu = document.createElement("select");
  rootMenu.setAttribute("aria-label", "Folder");
  styleField(rootMenu, "0 1 auto");
  rootMenu.style.maxWidth = "160px";

  const refreshChip = button("Refresh", () => {
    refresh();
  }, { kind: "tool", hint: "Read the folders again", logName });

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.multiple = true;
  fileInput.accept = KIND_ORDER.flatMap((kind) => KIND_EXTENSIONS[kind]).join(",");
  fileInput.style.cssText = "display:none";

  const uploadChip = button("Upload", () => {
    if (!uploading) fileInput.click();
  }, { kind: "tool", hint: "Add files to the input folder", logName });

  toolbar.append(search, chipRow, rootMenu, refreshChip, uploadChip, fileInput);

  const grid = document.createElement("div");
  grid.tabIndex = 0;
  grid.setAttribute("role", "listbox");
  grid.setAttribute("aria-label", "Files");
  grid.style.cssText = "flex:1 1 auto;min-height:0;overflow:auto;outline:none;"
    + `padding:${TILE_GAP + 2}px 12px;${SCROLLBARS}`;

  const cells = document.createElement("div");
  cells.style.cssText = "display:grid;"
    + `grid-template-columns:repeat(auto-fill,minmax(${thumbWidth}px,1fr));`
    + `gap:${TILE_GAP}px;align-content:start`;
  grid.appendChild(cells);

  const status = document.createElement("div");
  status.setAttribute("role", "status");
  status.style.cssText = "flex:0 0 auto;padding:6px 12px;white-space:nowrap;overflow:hidden;"
    + `text-overflow:ellipsis;border-top:1px solid ${LOOK.border};font-size:11px;`
    + `color:${LOOK.muted};background:${LOOK.surface}`;

  const menu = document.createElement("div");
  menu.setAttribute("role", "menu");
  menu.style.cssText = [
    "position:absolute",
    "z-index:2",
    "display:none",
    "min-width:150px",
    "max-width:260px",
    "padding:4px",
    FLOATING_PANEL,
  ].join(";");

  root.append(toolbar, grid, status, menu);

  /**
   * Write the status line.
   *
   * @param {string} text - What it says.
   * @param {boolean} [warning] - Whether it is drawn in the warning colour.
   * @returns {void}
   */
  function setStatus(text, warning = false) {
    status.textContent = text;
    status.style.color = warning ? LOOK.warning : LOOK.muted;
  }

  /**
   * Say how many files there are and how many are drawn.
   *
   * @returns {void}
   */
  function describe() {
    let text;
    if (!entries.length) text = "no files found";
    else if (!shown.length) text = "nothing matches";
    else if (passing > shown.length) {
      text = `${entries.length} files · showing the first ${shown.length} of ${passing}`;
    } else text = `${entries.length} files · showing ${shown.length}`;
    if (truncated) text += ` · listing cut at ${LISTING_LIMIT} files`;
    setStatus(text);
  }

  /**
   * Draw the kind chips for the kinds listed.
   *
   * @returns {void}
   */
  function renderChips() {
    chipRow.replaceChildren();
    for (const kind of kinds) {
      const chip = createKindPill(KIND_LABELS[kind], `Show ${KIND_LABELS[kind].toLowerCase()}`, () => {
        if (enabled.has(kind)) enabled.delete(kind);
        else enabled.add(kind);
        paintChip(chip, enabled.has(kind));
        applyFilter();
      });
      chip.textContent = KIND_LABELS[kind];
      paintChip(chip, enabled.has(kind));
      chipRow.appendChild(chip);
    }
  }

  /**
   * Draw the folder menu from the roots the listing named.
   *
   * @returns {void}
   */
  function renderRoots() {
    rootMenu.replaceChildren();
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "all folders";
    rootMenu.appendChild(all);
    for (const one of roots) {
      const option = document.createElement("option");
      option.value = one.tag;
      option.textContent = `${one.tag} (${one.files})`;
      rootMenu.appendChild(option);
    }
    if (rootTag && !roots.some((one) => one.tag === rootTag)) rootTag = "";
    rootMenu.value = rootTag;
  }

  /**
   * How many tiles share the first row.
   *
   * @returns {number} Columns, at least one.
   */
  function columnsOf() {
    if (!tiles.length) return 1;
    const top = tiles[0].offsetTop;
    let count = 1;
    while (count < tiles.length && tiles[count].offsetTop === top) count += 1;
    return count;
  }

  /**
   * Mark one tile as the selection.
   *
   * @param {number} index - The tile, or -1 for none.
   * @returns {void}
   */
  function select(index) {
    const previous = tiles[selected];
    if (previous && selected !== index) {
      previous.style.outline = "none";
      previous.setAttribute("aria-selected", "false");
      if (selected !== menuFor) pluses[selected].style.display = "none";
    }
    const tile = tiles[index];
    selected = tile ? index : -1;
    if (!tile) return;
    tile.style.outline = `2px solid ${LOOK.accent}`;
    tile.setAttribute("aria-selected", "true");
    pluses[index].style.display = "flex";
    tile.scrollIntoView?.({ block: "nearest" });
  }

  /**
   * Fill the menu with the roles for one tile.
   *
   * @param {number} index - The tile the menu is for.
   * @returns {void}
   */
  function fillMenu(index) {
    menu.replaceChildren();
    const heading = document.createElement("div");
    heading.textContent = "Use as";
    heading.style.cssText = `padding:5px 12px 4px;${LABEL}`;
    menu.appendChild(heading);
    // A role that names the kinds it takes is offered only to a file of one of them.
    const kind = shown[index]?.kind ?? null;
    const fitting = roles.filter((role) => !role.kinds || !kind || role.kinds.includes(kind));
    if (!fitting.length) {
      const none = document.createElement("div");
      none.textContent = kind ? `no role takes a ${kind}` : "no fitting role";
      none.style.cssText = `padding:7px 12px;color:${LOOK.muted}`;
      menu.appendChild(none);
      return;
    }
    for (const role of fitting) {
      const item = document.createElement("button");
      item.type = "button";
      item.setAttribute("role", "menuitem");
      item.textContent = role.label;
      item.style.cssText = [
        "display:block",
        "width:100%",
        "box-sizing:border-box",
        "padding:7px 12px",
        "border:0",
        "border-radius:6px",
        "background:none",
        "text-align:left",
        `font:13px/1.5 ${SANS}`,
        `color:${LOOK.text}`,
        "cursor:pointer",
        "white-space:nowrap",
        "overflow:hidden",
        "text-overflow:ellipsis",
      ].join(";");
      const light = () => {
        item.style.background = LOOK.hover;
      };
      const dim = () => {
        item.style.background = "none";
      };
      item.onmouseenter = light;
      item.onmouseleave = dim;
      item.onfocus = light;
      item.onblur = dim;
      item.addEventListener("click", (event) => {
        event.stopPropagation();
        choose(index, role.id);
      });
      menu.appendChild(item);
    }
  }

  /**
   * Put the menu beside one tile, inside the browser.
   *
   * @param {HTMLElement} tile - The tile.
   * @returns {void}
   */
  function placeMenu(tile) {
    const box = tile.getBoundingClientRect();
    const right = elementPoint(root, { clientX: box.right, clientY: box.top });
    const left = elementPoint(root, { clientX: box.left, clientY: box.top });
    const width = menu.offsetWidth;
    const height = menu.offsetHeight;
    let x = right.x + MENU_GAP;
    if (x + width > root.clientWidth) x = Math.max(0, left.x - width - MENU_GAP);
    let y = right.y;
    if (y + height > root.clientHeight) y = Math.max(0, root.clientHeight - height);
    menu.style.left = `${Math.round(x)}px`;
    menu.style.top = `${Math.round(y)}px`;
  }

  /**
   * Open the "Use as" menu beside one tile.
   *
   * @param {number} index - The tile.
   * @returns {void}
   */
  function openMenu(index) {
    const tile = tiles[index];
    if (!tile || disposed) return;
    if (menuFor >= 0) closeMenu(false);
    fillMenu(index);
    menuFor = index;
    pluses[index].style.display = "flex";
    menu.style.left = "0px";
    menu.style.top = "0px";
    menu.style.display = "block";
    placeMenu(tile);
    document.addEventListener("pointerdown", onDocumentPress, true);
    menu.querySelector("button")?.focus({ preventScroll: true });
  }

  /**
   * Close the "Use as" menu.
   *
   * @param {boolean} refocus - Whether the grid takes focus back.
   * @returns {void}
   */
  function closeMenu(refocus) {
    if (menuFor < 0) return;
    const was = menuFor;
    menuFor = -1;
    document.removeEventListener("pointerdown", onDocumentPress, true);
    menu.style.display = "none";
    menu.replaceChildren();
    if (pluses[was] && was !== selected) pluses[was].style.display = "none";
    if (refocus && !disposed) grid.focus({ preventScroll: true });
  }

  /**
   * Close the menu on a press outside it.
   *
   * @param {PointerEvent} event - The press.
   * @returns {void}
   */
  function onDocumentPress(event) {
    if (!menu.contains(event.target)) closeMenu(false);
  }

  /**
   * Hand one tile's entry to the caller for one role.
   *
   * @param {number} index - The tile.
   * @param {string} roleId - The role picked.
   * @returns {void}
   */
  function choose(index, roleId) {
    const entry = shown[index];
    closeMenu(true);
    if (!entry || !onChoose) return;
    try {
      onChoose({ ...entry }, roleId);
    } catch (error) {
      console.error(`[${logName}] The choose handler failed:`, error);
    }
  }

  /**
   * One tile.
   *
   * @param {object} entry - The entry it draws.
   * @param {number} index - Its place in the grid.
   * @returns {{tile: HTMLElement, plus: HTMLButtonElement}} The tile and its "+" button.
   */
  function createTile(entry, index) {
    const tile = document.createElement("div");
    tile.setAttribute("role", "option");
    tile.setAttribute("aria-selected", "false");
    tile.tabIndex = -1;
    const when = dateOf(entry.mtime);
    tile.title = `${entry.label}\n${readableBytes(entry.size)}${when ? ` · ${when}` : ""}`;
    tile.style.cssText = "display:flex;flex-direction:column;gap:4px;min-width:0;padding:5px;"
      + `border-radius:8px;cursor:pointer;outline:none;outline-offset:-1px;`
      + `background:${LOOK.surface};border:1px solid ${LOOK.border}`;

    const thumb = document.createElement("div");
    thumb.style.cssText = [
      "position:relative",
      "width:100%",
      `height:${thumbHeight}px`,
      "border-radius:5px",
      "overflow:hidden",
      "display:flex",
      "align-items:center",
      "justify-content:center",
      `background:${LOOK.body}`,
      `color:${LOOK.muted}`,
    ].join(";");

    if (entry.kind === "sound") {
      thumb.appendChild(createGlyph("sound"));
    } else {
      const image = document.createElement("img");
      image.loading = "lazy";
      image.decoding = "async";
      image.draggable = false;
      image.alt = "";
      image.style.cssText = "display:block;width:100%;height:100%;object-fit:cover";
      const fallBack = () => {
        image.remove();
        thumb.appendChild(createGlyph(entry.kind === "clip" ? "clip" : "broken"));
      };
      image.onerror = fallBack;
      image.onload = () => {
        if (!image.naturalWidth) fallBack();
      };
      image.src = thumbnailUrl(entry);
      thumb.appendChild(image);
    }

    const badge = document.createElement("span");
    badge.textContent = entry.kind;
    badge.style.cssText = "position:absolute;left:4px;top:4px;padding:0 6px;border-radius:999px;"
      + "font-size:9px;font-weight:600;line-height:15px;letter-spacing:0.05em;text-transform:uppercase;"
      + `background:${LOOK.surface};color:${LOOK.muted};border:1px solid ${LOOK.border};pointer-events:none`;

    const plus = document.createElement("button");
    plus.type = "button";
    plus.title = "Use this file as";
    plus.setAttribute("aria-label", `Use ${baseName(entry)} as`);
    plus.textContent = "+";
    plus.style.cssText = [
      "position:absolute",
      "right:4px",
      "bottom:4px",
      "width:20px",
      "height:20px",
      "padding:0",
      "display:none",
      "align-items:center",
      "justify-content:center",
      "border-radius:5px",
      `font:600 14px/1 ${SANS}`,
      "cursor:pointer",
      `border:1px solid ${LOOK.border}`,
      `background:${LOOK.surface}`,
      `color:${LOOK.text}`,
    ].join(";");
    plus.addEventListener("click", (event) => {
      event.stopPropagation();
      select(index);
      openMenu(index);
    });
    thumb.append(badge, plus);

    const name = document.createElement("div");
    name.textContent = baseName(entry);
    name.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:0 2px;"
      + `font-size:12px;font-weight:600;color:${LOOK.text}`;

    const where = document.createElement("div");
    where.textContent = entry.tag;
    where.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:0 2px;"
      + `font-size:11px;color:${LOOK.muted}`;

    tile.append(thumb, name, where);
    tile.onmouseenter = () => {
      plus.style.display = "flex";
      tile.style.background = LOOK.hover;
    };
    tile.onmouseleave = () => {
      tile.style.background = LOOK.surface;
      if (index !== selected && index !== menuFor) plus.style.display = "none";
    };
    tile.addEventListener("click", () => {
      select(index);
      openMenu(index);
      try {
        onSelect?.({ ...entry });
      } catch (error) {
        console.error(`[${logName}] Failed to answer a tile press:`, error);
      }
    });
    tile.draggable = true;
    tile.addEventListener("dragstart", (event) => {
      closeMenu(false);
      try {
        event.dataTransfer.setData(DRAG_TYPE, entry.label);
        event.dataTransfer.effectAllowed = "copy";
        const picture = thumb.querySelector("img");
        if (picture?.naturalWidth) event.dataTransfer.setDragImage(picture, 24, 18);
      } catch (error) {
        console.error(`[${logName}] Failed to start a drag:`, error);
      }
      tile.style.opacity = "0.5";
      try {
        onDragStart?.({ ...entry });
      } catch (error) {
        console.error(`[${logName}] The drag start handler failed:`, error);
      }
    });
    tile.addEventListener("dragend", () => {
      tile.style.opacity = "";
      try {
        onDragEnd?.();
      } catch (error) {
        console.error(`[${logName}] The drag end handler failed:`, error);
      }
    });
    return { tile, plus };
  }

  /**
   * Draw the tiles for the entries that pass the filter.
   *
   * @returns {void}
   */
  function renderTiles() {
    closeMenu(false);
    const keep = selected >= 0 ? shown[selected] : null;
    const keptLabel = keep ? keep.label : "";
    selected = -1;
    tiles = [];
    pluses = [];
    const fragment = document.createDocumentFragment();
    shown.forEach((entry, index) => {
      const { tile, plus } = createTile(entry, index);
      tiles.push(tile);
      pluses.push(plus);
      fragment.appendChild(tile);
    });
    cells.replaceChildren(fragment);
    const again = keptLabel ? shown.findIndex((entry) => entry.label === keptLabel) : -1;
    if (again >= 0) select(again);
    describe();
  }

  /**
   * Keep the entries matching the search, the chips and the folder, and draw them.
   *
   * @returns {void}
   */
  function applyFilter() {
    const needle = query.trim().toLowerCase();
    const kept = entries.filter((entry) => enabled.has(entry.kind)
      && (!rootTag || entry.tag === rootTag)
      && (!needle || entry.label.toLowerCase().includes(needle)));
    passing = kept.length;
    shown = kept.slice(0, TILE_CAP);
    renderTiles();
  }

  /**
   * Take a listing the route answered.
   *
   * @param {object} answer - `{roots, entries, truncated}` as the route answers it.
   * @returns {void}
   */
  function take(answer) {
    const rows = Array.isArray(answer?.entries) ? answer.entries : [];
    entries = [];
    for (const row of rows) {
      const kind = kindOf(row);
      const label = String(row?.label ?? "");
      if (!kind || !label) continue;
      entries.push({
        label,
        relative: String(row.relative ?? ""),
        tag: String(row.tag ?? ""),
        size: Number(row.size) || 0,
        mtime: Number(row.mtime) || 0,
        kind,
      });
    }
    const named = Array.isArray(answer?.roots) ? answer.roots : [];
    roots = named.map((one) => ({ tag: String(one?.tag ?? ""), files: Number(one?.files) || 0 }))
      .filter((one) => one.tag);
    truncated = answer?.truncated === true;
    renderRoots();
    applyFilter();
  }

  /**
   * Read the listing again.
   *
   * @returns {Promise<void>} Settles when the grid has been redrawn or the read has failed.
   */
  async function refresh() {
    if (disposed) return;
    controller?.abort();
    const mine = new AbortController();
    controller = mine;
    setStatus("reading the folders");
    const ext = kinds.flatMap((kind) => KIND_EXTENSIONS[kind]).join(",");
    const route = `${LISTING_ROUTE}?ext=${encodeURIComponent(ext)}&limit=${LISTING_LIMIT}`;
    try {
      const response = await fetchWithin(route, { cache: "no-store", signal: mine.signal });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const answer = await response.json();
      if (mine.signal.aborted || disposed) return;
      take(answer);
    } catch (error) {
      if (mine.signal.aborted || disposed) return;
      console.error(`[${logName}] Failed to read the file listing:`, error);
      setStatus("the file list could not be read", true);
    } finally {
      if (controller === mine) controller = null;
    }
  }

  /**
   * Send the pictures, clips and sounds among some files to the input folder, then read the
   * listing again.
   *
   * @param {File[]} chosen - What to send. Files of any other kind are left out.
   * @returns {Promise<object[]>} One `{label, kind}` per file sent, once every file has been
   *   sent or refused.
   */
  async function upload(chosen) {
    if (!chosen.length || uploading || disposed) return [];
    const files = chosen.filter((file) => kindOf({ relative: file.name }));
    if (!files.length) {
      setStatus("only pictures, clips and sounds can be uploaded", true);
      return [];
    }
    uploading = true;
    uploadChip.disabled = true;
    const failed = chosen.filter((file) => !files.includes(file)).map((file) => file.name);
    const skipped = failed.length;
    const placed = [];
    let sent = 0;
    for (const file of files) {
      if (disposed) return placed;
      setStatus(`uploading ${file.name} (${sent + failed.length - skipped + 1} of ${files.length})`);
      try {
        const body = new FormData();
        body.append("image", file, file.name);
        body.append("type", "input");
        body.append("subfolder", "");
        body.append("overwrite", "false");
        const response = await fetchWithin(
          UPLOAD_ROUTE, { method: "POST", body, cache: "no-store" }, UPLOAD_TIMEOUT,
        );
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        const label = uploadedLabel(await response.json());
        placed.push({ label, kind: kindOf({ label }) });
        sent += 1;
      } catch (error) {
        console.error(`[${logName}] Failed to upload ${file.name}:`, error);
        failed.push(file.name);
      }
    }
    uploading = false;
    uploadChip.disabled = false;
    await refresh();
    if (disposed || !failed.length) return placed;
    setStatus(failed.length === 1
      ? `${failed[0]} could not be uploaded`
      : `${failed.length} files could not be uploaded`, true);
    return placed;
  }

  /**
   * Replace the roles the menu offers.
   *
   * @param {Array<{id: string, label: string}>} next - The roles.
   * @returns {void}
   */
  function setRoles(next) {
    roles = normaliseRoles(next);
    if (menuFor >= 0) {
      const index = menuFor;
      fillMenu(index);
      placeMenu(tiles[index]);
    }
  }

  /**
   * Replace the kinds listed, and read the listing again.
   *
   * @param {string[]} next - Any of `picture`, `clip` and `sound`.
   * @returns {void}
   */
  function setKinds(next) {
    kinds = normaliseKinds(next);
    enabled = new Set(kinds);
    renderChips();
    refresh();
  }

  /**
   * Abandon the read in flight, release the page listeners and empty the element.
   *
   * @returns {void}
   */
  function dispose() {
    if (disposed) return;
    closeMenu(false);
    disposed = true;
    clearTimeout(searchTimer);
    controller?.abort();
    controller = null;
    root.replaceChildren();
  }

  search.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      if (disposed) return;
      query = search.value;
      applyFilter();
    }, SEARCH_SETTLE_MS);
  });

  rootMenu.addEventListener("change", () => {
    rootTag = rootMenu.value;
    applyFilter();
  });

  fileInput.addEventListener("change", () => {
    const files = Array.from(fileInput.files ?? []);
    fileInput.value = "";
    upload(files);
  });

  grid.addEventListener("scroll", () => closeMenu(false), { passive: true });

  if (acceptFiles) {
    const lit = (on) => {
      grid.style.outline = on ? `2px dashed ${LOOK.accent}` : "none";
      grid.style.outlineOffset = on ? "-6px" : "";
    };
    grid.addEventListener("dragover", (event) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      event.stopPropagation();
      event.dataTransfer.dropEffect = "copy";
      lit(true);
    });
    grid.addEventListener("dragleave", (event) => {
      if (!grid.contains(event.relatedTarget)) lit(false);
    });
    grid.addEventListener("drop", (event) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      event.stopPropagation();
      lit(false);
      upload(Array.from(event.dataTransfer.files ?? []));
    });
  }

  grid.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const count = tiles.length;
    if (!count) return;
    let next = selected;
    switch (event.key) {
      case "ArrowRight":
        next = Math.min(count - 1, selected + 1);
        break;
      case "ArrowLeft":
        next = Math.max(0, selected - 1);
        break;
      case "ArrowDown":
        next = selected < 0 ? 0 : Math.min(count - 1, selected + columnsOf());
        break;
      case "ArrowUp":
        next = selected < 0 ? 0 : Math.max(0, selected - columnsOf());
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = count - 1;
        break;
      case "Enter":
      case " ":
        if (selected < 0) return;
        event.preventDefault();
        event.stopPropagation();
        openMenu(selected);
        return;
      default:
        return;
    }
    event.preventDefault();
    event.stopPropagation();
    select(next);
  });

  menu.addEventListener("keydown", (event) => {
    const items = Array.from(menu.querySelectorAll("button"));
    if (!items.length) return;
    const at = items.indexOf(document.activeElement);
    let next;
    if (event.key === "ArrowDown") next = at < 0 ? 0 : (at + 1) % items.length;
    else if (event.key === "ArrowUp") next = at <= 0 ? items.length - 1 : at - 1;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = items.length - 1;
    else return;
    event.preventDefault();
    event.stopPropagation();
    items[next].focus({ preventScroll: true });
  });

  root.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || menuFor < 0) return;
    event.preventDefault();
    event.stopPropagation();
    closeMenu(true);
  });

  renderChips();
  renderRoots();
  describe();
  refresh();

  return { element: root, refresh, setRoles, setKinds, upload, dispose };
}
