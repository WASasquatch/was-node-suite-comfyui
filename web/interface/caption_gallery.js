/**
 * The gallery of pictures Image Save wrote, each caption shown over its picture.
 *
 * Answers the `{element, update, clear, dispose, refresh}` a report panel's `sketch` takes.
 * Thumbnails are fetched by menu label; a click opens one at full size.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { CARD_LOOK, cardChip } from "./card.js";
import { executionId, nodeLocator, ownOutput } from "./preview.js";
import { openLightbox } from "./lightbox.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { themeVar } from "./theme.js";
import { chainWidgetCallback, keepOnNode } from "./widget.js";
import { SCROLLBARS } from "./window_kit.js";

const LOG_NAME = "WASNodeSuite.CaptionGallery";

const THUMB_ROUTE = "/was/interface/api/file_thumbnail";
const VIEW_ROUTE = "/view";

// The roots ComfyUI's own file route serves, as the tags a label carries.
const VIEW_TYPES = new Set(["input", "output", "temp"]);

// Run output key the node writes the gallery under.
const OUTPUT_KEY = "was_gallery";

// The widget choosing the gallery, and the choice this band draws for.
const MODE_WIDGET = "gallery";
const MODE = "interface gallery";

// The widths a picture can be drawn at, in CSS pixels, and the one a new node starts at.
const TILE_SIZES = [64, 96, 128, 176, 240, 320];
const DEFAULT_TILE = 128;

// The node property holding the width chosen on this node.
const TILE_PROPERTY = "was_gallery_tile";

// The edges a thumbnail is fetched at, smallest first, and the largest the route answers.
const THUMB_EDGES = [192, 320, 512];
const LARGEST_THUMB = 512;

// The shortest the band is drawn, and the height of the caption strip under the pictures.
const BAND_HEIGHT = 180;
const STRIP_HEIGHT = 58;

// The scrim a caption is written on, and the text on it, over any picture in either theme.
const SCRIM = "linear-gradient(to top, rgba(0,0,0,0.88), rgba(0,0,0,0.62))";
const SCRIM_TEXT = "#f2f2f2";

const EMPTY_LABEL = "run the node to see the pictures it saves";
const NO_CAPTION = "no caption";

/**
 * The thumbnail address of one saved picture.
 *
 * @param {object} item - A gallery entry carrying `label` and `stamp`.
 * @param {number} edge - Pixels a side the thumbnail is fitted inside.
 * @returns {string} The address, or `""` for a file no listed folder holds.
 */
function thumbnailUrl(item, edge) {
  if (!item?.label) return "";
  const query = new URLSearchParams({
    label: String(item.label),
    edge: String(edge),
    stamp: String(item.stamp ?? ""),
  });
  return api.apiURL(`${THUMB_ROUTE}?${query.toString()}`);
}

/**
 * The thumbnail edge that covers a tile at the density the grid is drawn at.
 *
 * @param {number} tile - The tile's width in CSS pixels.
 * @param {HTMLElement} grid - The element the tiles sit in.
 * @returns {number} One of `THUMB_EDGES`.
 */
function edgeFor(tile, grid) {
  const wanted = tile * Math.min(2, surfaceRatio(grid));
  return THUMB_EDGES.find((edge) => edge >= wanted) ?? LARGEST_THUMB;
}

/**
 * The address of one saved picture at full size, or the largest thumbnail where ComfyUI does
 * not serve its folder.
 *
 * @param {object} item - A gallery entry carrying `label` and `stamp`.
 * @returns {{url: string, note?: string}} The address, and what it is when not the whole file.
 */
function fullSource(item) {
  const match = /^(.*) \[([^\]]+)\]$/.exec(String(item?.label ?? ""));
  if (match && VIEW_TYPES.has(match[2])) {
    const relative = match[1];
    const cut = relative.lastIndexOf("/");
    const query = new URLSearchParams({
      filename: cut >= 0 ? relative.slice(cut + 1) : relative,
      subfolder: cut >= 0 ? relative.slice(0, cut) : "",
      type: match[2],
      rand: String(item.stamp ?? ""),
    });
    return { url: api.apiURL(`${VIEW_ROUTE}?${query.toString()}`) };
  }
  const url = thumbnailUrl(item, LARGEST_THUMB);
  return url ? { url, note: `preview at ${LARGEST_THUMB} px` } : { url: "" };
}

/**
 * The tile width this node was left at.
 *
 * @param {object} node - The Image Save node.
 * @returns {number} One of `TILE_SIZES`.
 */
function tileOf(node) {
  const stored = Number(node?.properties?.[TILE_PROPERTY]);
  return TILE_SIZES.includes(stored) ? stored : DEFAULT_TILE;
}

/**
 * Build the gallery band.
 *
 * @param {object} node - The Image Save node the panel is on.
 * @returns {{element: HTMLElement, update: Function, clear: Function, dispose: Function,
 *   refresh: Function}} The band, in the shape a report panel's sketch takes.
 */
export function createCaptionGallery(node) {
  const band = document.createElement("div");
  band.className = "was-caption-gallery";
  // Takes whatever height the node leaves it, so the grid scrolls rather than growing the node.
  band.style.cssText = "display:flex;flex-direction:column;gap:4px;flex:3 1 0;"
    + `min-height:${BAND_HEIGHT}px;overflow:hidden`;

  const head = document.createElement("div");
  head.style.cssText = "flex:none;display:flex;align-items:center;gap:6px;min-width:0";
  const title = document.createElement("span");
  title.style.cssText = "font-variant:small-caps;letter-spacing:0.05em;font-weight:650;"
    + `color:${CARD_LOOK.title}`;
  title.textContent = "gallery";
  const chips = document.createElement("span");
  chips.style.cssText = "display:flex;gap:4px;min-width:0;overflow:hidden";
  // The tile width, stepped from the head of the band.
  const sizer = document.createElement("span");
  sizer.style.cssText = "margin-left:auto;flex:none;display:flex;align-items:center;gap:2px";
  const sizeButton = (mark, step, hint) => {
    const control = document.createElement("button");
    control.type = "button";
    control.textContent = mark;
    control.title = hint;
    control.style.cssText = "width:20px;height:18px;padding:0;border-radius:4px;cursor:pointer;"
      + `font:13px/16px system-ui,sans-serif;color:${CARD_LOOK.text};background:${CARD_LOOK.inset};`
      + `border:1px solid ${CARD_LOOK.edge}`;
    control.addEventListener("pointerdown", (event) => event.stopPropagation());
    control.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      stepTile(step);
    });
    return control;
  };
  const smaller = sizeButton("\u2212", -1, "Smaller pictures");
  const sizeLabel = document.createElement("span");
  sizeLabel.style.cssText = "min-width:40px;text-align:center;font-variant-numeric:tabular-nums;"
    + `color:${themeVar("fgMuted")}`;
  const larger = sizeButton("+", 1, "Larger pictures");
  sizer.append(smaller, sizeLabel, larger);
  head.append(title, chips, sizer);

  const well = document.createElement("div");
  well.style.cssText = "position:relative;flex:1 1 0;min-height:0;overflow-x:hidden;overflow-y:auto;"
    + "scrollbar-gutter:stable;"
    + `border-radius:6px;padding:6px;background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};`
    + SCROLLBARS;
  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;gap:6px";
  const empty = document.createElement("div");
  empty.style.cssText = `padding:8px 4px;color:${themeVar("fgMuted")}`;
  empty.textContent = EMPTY_LABEL;
  well.append(empty, grid);

  // The whole caption of the picture under the pointer, or the one last clicked.
  const strip = document.createElement("div");
  strip.style.cssText = `flex:none;height:${STRIP_HEIGHT}px;overflow:auto;border-radius:6px;`
    + `padding:4px 6px;background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};`
    + "white-space:pre-wrap;overflow-wrap:anywhere";
  const stripName = document.createElement("div");
  stripName.style.cssText = `color:${themeVar("fgMuted")};font-size:9px;overflow:hidden;`
    + "text-overflow:ellipsis;white-space:nowrap";
  const stripText = document.createElement("div");
  strip.append(stripName, stripText);

  band.append(head, well, strip);

  let report = null;
  let pinned = -1;
  let disposed = false;
  let tile = tileOf(node);
  // The thumbnail edge the tiles on screen were fetched at.
  let drawnEdge = null;
  const tiles = [];

  /**
   * Draw the grid at the node's tile width, and fetch thumbnails to match.
   *
   * @returns {boolean} True when the thumbnail edge changed, so the tiles need fetching again.
   */
  const applyTile = () => {
    const before = edgeFor(tile, grid);
    tile = tileOf(node);
    grid.style.gridTemplateColumns = `repeat(auto-fill,minmax(${tile}px,1fr))`;
    sizeLabel.textContent = `${tile} px`;
    const index = TILE_SIZES.indexOf(tile);
    smaller.disabled = index <= 0;
    larger.disabled = index >= TILE_SIZES.length - 1;
    smaller.style.opacity = smaller.disabled ? "0.4" : "1";
    larger.style.opacity = larger.disabled ? "0.4" : "1";
    return edgeFor(tile, grid) !== before;
  };

  /**
   * Step the tile width and keep it on the node.
   *
   * @param {number} step - Steps to move through `TILE_SIZES`, negative for smaller.
   * @returns {void}
   */
  const stepTile = (step) => {
    const index = TILE_SIZES.indexOf(tileOf(node));
    const next = TILE_SIZES[Math.max(0, Math.min(TILE_SIZES.length - 1, index + step))];
    if (next === tileOf(node)) return;
    keepOnNode(node, TILE_PROPERTY, next);
    if (applyTile()) render();
  };

  /**
   * Open one picture at full size, its caption beside it.
   *
   * @param {number} index - The picture.
   * @returns {void}
   */
  const open = (index) => {
    const items = Array.isArray(report?.items) ? report.items : [];
    if (!items[index]) return;
    openLightbox({
      items,
      index,
      source: fullSource,
      onIndex: (shown) => {
        pinned = shown;
        markPinned();
        describe(pinned);
      },
    });
  };

  /**
   * Write one picture's name and caption into the strip, or the strip's resting words.
   *
   * @param {number} index - The picture, or -1 for none.
   * @returns {void}
   */
  const describe = (index) => {
    const item = report?.items?.[index];
    if (!item) {
      stripName.textContent = "";
      stripText.textContent = report?.items?.length
        ? "hover to read a caption, click to open the picture"
        : "";
      stripText.style.color = themeVar("fgMuted");
      return;
    }
    stripName.textContent = item.name ?? "";
    stripText.textContent = item.caption || NO_CAPTION;
    stripText.style.color = item.caption ? CARD_LOOK.text : themeVar("fgMuted");
  };

  /**
   * Outline the clicked picture and leave the rest plain.
   *
   * @returns {void}
   */
  const markPinned = () => {
    tiles.forEach((box, index) => {
      box.style.outline = index === pinned ? `2px solid ${CARD_LOOK.rule}` : "none";
    });
  };

  /**
   * One picture with its caption laid over it while the pointer is on it.
   *
   * @param {object} item - The gallery entry.
   * @param {number} index - Its place in the gallery.
   * @returns {HTMLElement} The tile.
   */
  const buildTile = (item, index) => {
    const box = document.createElement("div");
    box.style.cssText = "position:relative;aspect-ratio:1;border-radius:4px;overflow:hidden;"
      + `background:${CARD_LOOK.inset};cursor:zoom-in;outline-offset:-2px`;
    box.title = "Open at full size";
    const url = thumbnailUrl(item, edgeFor(tile, grid));
    if (url) {
      const picture = document.createElement("img");
      picture.loading = "lazy";
      picture.decoding = "async";
      picture.draggable = false;
      picture.alt = item.name ?? "";
      picture.src = url;
      picture.style.cssText = "display:block;width:100%;height:100%;object-fit:cover";
      box.appendChild(picture);
    } else {
      const name = document.createElement("div");
      name.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;"
        + `justify-content:center;padding:4px;text-align:center;color:${themeVar("fgMuted")};`
        + "font-size:9px;overflow-wrap:anywhere";
      name.textContent = item.name ?? "";
      box.appendChild(name);
    }
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:absolute;inset:0;padding:5px 6px;opacity:0;"
      + `transition:opacity 120ms ease;background:${SCRIM};color:${SCRIM_TEXT};`
      + "font-size:9.5px;line-height:1.3;overflow:hidden;overflow-wrap:anywhere;"
      + "pointer-events:none;-webkit-mask-image:linear-gradient(to bottom,#000 78%,transparent);"
      + "mask-image:linear-gradient(to bottom,#000 78%,transparent)";
    overlay.textContent = item.caption || item.name || NO_CAPTION;
    if (!item.caption) overlay.style.fontStyle = "italic";
    box.appendChild(overlay);

    box.addEventListener("pointerenter", () => {
      overlay.style.opacity = "1";
      describe(index);
    });
    box.addEventListener("pointerleave", () => {
      overlay.style.opacity = "0";
      describe(pinned);
    });
    box.addEventListener("click", () => {
      pinned = index;
      markPinned();
      describe(pinned);
      open(index);
    });
    return box;
  };

  /**
   * Whether the node is set to draw this gallery.
   *
   * @returns {boolean} True for the interface gallery, and where the widget cannot be read.
   */
  const chosen = () => {
    const widget = (node.widgets ?? []).find((candidate) => candidate?.name === MODE_WIDGET);
    return !widget || widget.value === MODE;
  };

  /**
   * Draw the gallery the node last wrote, or hide the band when another gallery is chosen.
   *
   * @returns {void}
   */
  const render = () => {
    if (disposed) return;
    band.style.display = chosen() ? "flex" : "none";
    applyTile();
    drawnEdge = edgeFor(tile, grid);
    grid.textContent = "";
    tiles.length = 0;
    chips.textContent = "";
    const items = Array.isArray(report?.items) ? report.items : [];
    empty.style.display = items.length ? "none" : "";
    if (report) {
      const total = Number(report.total) || items.length;
      chips.appendChild(cardChip(
        items.length < total ? `${items.length} of ${total}` : `${total} pictures`,
        "accent",
      ));
      const captioned = items.filter((item) => item?.caption).length;
      if (captioned) chips.appendChild(cardChip(`${captioned} captioned`));
      if (report.history) chips.appendChild(cardChip("history"));
      if (!items.length) empty.textContent = report.history ? "nothing written before" : "nothing saved";
    } else {
      empty.textContent = EMPTY_LABEL;
    }
    const fragment = document.createDocumentFragment();
    items.forEach((item, index) => {
      const box = buildTile(item, index);
      tiles.push(box);
      fragment.appendChild(box);
    });
    grid.appendChild(fragment);
    if (pinned >= items.length) pinned = -1;
    markPinned();
    describe(pinned);
    node.setDirtyCanvas?.(true, false);
  };

  /**
   * Take the gallery a run output carries, when this workflow's run wrote it.
   *
   * @param {object} output - A node's run output.
   * @returns {boolean} True when a gallery was taken.
   */
  const take = (output) => {
    const fresh = output?.[OUTPUT_KEY]?.[0];
    if (!fresh || !ownOutput(fresh)) return false;
    if (report !== fresh) pinned = -1;
    report = fresh;
    return true;
  };

  const refresh = () => {
    try {
      const store = app?.nodeOutputs ?? {};
      take(store[nodeLocator(node)] ?? store[executionId(node)]);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the last gallery:`, error);
    }
    render();
  };

  const onExecuted = (event) => {
    try {
      const detail = event?.detail;
      const shown = String(detail?.display_node ?? detail?.node ?? "");
      if (!shown || shown !== executionId(node)) return;
      // A run drawing another gallery leaves nothing of this one standing.
      if (!take(detail?.output)) report = null;
      render();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the node's gallery:`, error);
    }
  };
  api.addEventListener("executed", onExecuted);

  chainWidgetCallback(node, MODE_WIDGET, render, LOG_NAME);

  refresh();

  // A zoom that changes the drawn density fetches thumbnails at the edge it now needs.
  const stopRatio = watchSurfaceRatio(grid, () => {
    if (!disposed && drawnEdge !== null && edgeFor(tile, grid) !== drawnEdge) render();
  });

  return {
    element: band,
    update: () => {
      band.style.display = chosen() ? "flex" : "none";
    },
    clear: () => {},
    refresh,
    dispose() {
      if (disposed) return;
      disposed = true;
      stopRatio();
      api.removeEventListener?.("executed", onExecuted);
      grid.textContent = "";
      tiles.length = 0;
    },
  };
}
