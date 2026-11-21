/**
 * The gallery of pictures Image Save wrote, each caption shown over its picture.
 *
 * Answers the `{element, update, clear, dispose, refresh}` a report panel's `sketch` takes.
 * Pictures are thumbnails fetched by menu label.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { CARD_LOOK, cardChip } from "./card.js";
import { executionId, nodeLocator, ownOutput } from "./preview.js";
import { themeVar } from "./theme.js";
import { chainWidgetCallback } from "./widget.js";

const LOG_NAME = "WASNodeSuite.CaptionGallery";

const THUMB_ROUTE = "/was/interface/api/file_thumbnail";

// Run output key the node writes the gallery under.
const OUTPUT_KEY = "was_gallery";

// The widget choosing the gallery, and the choice this band draws for.
const MODE_WIDGET = "gallery";
const MODE = "interface gallery";

// Narrowest a picture is drawn, and the edge its thumbnail is fetched at, in CSS pixels.
const TILE = 92;
const THUMB_EDGE = 192;

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
 * @returns {string} The address, or `""` for a file no listed folder holds.
 */
function thumbnailUrl(item) {
  if (!item?.label) return "";
  const query = new URLSearchParams({
    label: String(item.label),
    edge: String(THUMB_EDGE),
    stamp: String(item.stamp ?? ""),
  });
  return api.apiURL(`${THUMB_ROUTE}?${query.toString()}`);
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
  band.style.cssText = "display:flex;flex-direction:column;gap:4px;flex:3 1 auto;"
    + `min-height:${BAND_HEIGHT}px;overflow:hidden`;

  const head = document.createElement("div");
  head.style.cssText = "flex:none;display:flex;align-items:center;gap:6px;min-width:0";
  const title = document.createElement("span");
  title.style.cssText = "font-variant:small-caps;letter-spacing:0.05em;font-weight:650;"
    + `color:${CARD_LOOK.title}`;
  title.textContent = "gallery";
  const chips = document.createElement("span");
  chips.style.cssText = "display:flex;gap:4px;min-width:0;overflow:hidden";
  head.append(title, chips);

  const well = document.createElement("div");
  well.style.cssText = "position:relative;flex:1 1 auto;min-height:0;overflow:auto;"
    + `border-radius:6px;padding:6px;background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well}`;
  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;gap:6px;"
    + `grid-template-columns:repeat(auto-fill,minmax(${TILE}px,1fr))`;
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
  const tiles = [];

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
      stripText.textContent = report?.items?.length ? "hover to read a caption" : "";
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
    tiles.forEach((tile, index) => {
      tile.style.outline = index === pinned ? `2px solid ${CARD_LOOK.rule}` : "none";
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
    const tile = document.createElement("div");
    tile.style.cssText = "position:relative;aspect-ratio:1;border-radius:4px;overflow:hidden;"
      + `background:${CARD_LOOK.inset};cursor:pointer;outline-offset:-2px`;
    const url = thumbnailUrl(item);
    if (url) {
      const picture = document.createElement("img");
      picture.loading = "lazy";
      picture.decoding = "async";
      picture.draggable = false;
      picture.alt = item.name ?? "";
      picture.src = url;
      picture.style.cssText = "display:block;width:100%;height:100%;object-fit:cover";
      tile.appendChild(picture);
    } else {
      const name = document.createElement("div");
      name.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;"
        + `justify-content:center;padding:4px;text-align:center;color:${themeVar("fgMuted")};`
        + "font-size:9px;overflow-wrap:anywhere";
      name.textContent = item.name ?? "";
      tile.appendChild(name);
    }
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:absolute;inset:0;padding:5px 6px;opacity:0;"
      + `transition:opacity 120ms ease;background:${SCRIM};color:${SCRIM_TEXT};`
      + "font-size:9.5px;line-height:1.3;overflow:hidden;overflow-wrap:anywhere;"
      + "pointer-events:none;-webkit-mask-image:linear-gradient(to bottom,#000 78%,transparent);"
      + "mask-image:linear-gradient(to bottom,#000 78%,transparent)";
    overlay.textContent = item.caption || item.name || NO_CAPTION;
    if (!item.caption) overlay.style.fontStyle = "italic";
    tile.appendChild(overlay);

    tile.addEventListener("pointerenter", () => {
      overlay.style.opacity = "1";
      describe(index);
    });
    tile.addEventListener("pointerleave", () => {
      overlay.style.opacity = "0";
      describe(pinned);
    });
    tile.addEventListener("click", () => {
      pinned = pinned === index ? -1 : index;
      markPinned();
      describe(pinned >= 0 ? pinned : index);
    });
    return tile;
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
      const tile = buildTile(item, index);
      tiles.push(tile);
      fragment.appendChild(tile);
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
      api.removeEventListener?.("executed", onExecuted);
      grid.textContent = "";
      tiles.length = 0;
    },
  };
}
