/**
 * What Image Vectorscope and Image False Color drew, shown on the node.
 *
 * Each draws its first frame's published picture; false colour adds a legend of every band's
 * colour, levels and share, read from its report.
 */

import { app } from "../../scripts/app.js";
import { CARD_LOOK } from "./interface/card.mjs";
import { createPictureBand } from "./interface/picture_band.mjs";
import { createReportPanel } from "./interface/report_panel.mjs";
import { themeVar } from "./interface/theme.mjs";
import { appendInterfaceWidget } from "./interface/widget.mjs";

const EXT_NAME = "WASNodeSuite.ImageScopesUI";
const VECTORSCOPE = "WASImageVectorscope";
const FALSE_COLOR = "WASImageFalseColor";
const NODE_IDS = new Set([VECTORSCOPE, FALSE_COLOR]);

// The output slots the two nodes publish their first frame under.
const SCOPE_SLOT = "vectorscope";
const FALSE_COLOR_SLOT = "false_color";

// Heights in node units the panels open at, and of the picture inside each, in CSS pixels.
const SCOPE_PANEL_HEIGHT = 380;
const SCOPE_PICTURE_HEIGHT = 240;
const FALSE_COLOR_PANEL_HEIGHT = 400;
const FALSE_COLOR_PICTURE_HEIGHT = 170;

// The narrowest each panel stays readable in; the legend needs four columns.
const PANEL_MIN_WIDTH = 270;
const FALSE_COLOR_MIN_WIDTH = 330;

const UI_WIDGET_NAME = "was_image_scopes_ui";
const UI_WIDGET_TYPE = "was_image_scopes";

// The report entries the legend reads its colours and levels from.
const COLOURS_FACT = "colours";
const LEVELS_FACT = "levels";

// A band colour is drawn only when it is a plain six digit hex code.
const HEX_COLOUR = /^#[0-9a-f]{6}$/i;

/**
 * The value of one fact a report carries.
 *
 * @param {object|null} report - A report from `run_result`.
 * @param {string} name - The fact's name.
 * @returns {string} Its value, or an empty string.
 */
function factOf(report, name) {
  const found = (report?.facts ?? []).find((entry) => entry.name === name);
  return found ? String(found.value ?? "") : "";
}

/**
 * A band's signal levels in words.
 *
 * @param {string} span - `low-high` in percent, as the node writes it.
 * @param {number} index - The band's place in the legend.
 * @param {number} count - How many bands there are.
 * @returns {string} The range, with the open ends of the first and last band said as such.
 */
function rangeText(span, index, count) {
  const [low, high] = String(span ?? "").split("-");
  if (!low || !high) return "";
  if (index === 0) return `under ${high}%`;
  if (index === count - 1) return `${low}% and up`;
  return `${low} to ${high}%`;
}

/**
 * Build the false colour picture with its legend under it.
 *
 * @param {object} node - The node the band is drawn on.
 * @returns {{element: HTMLElement, update: Function, clear: Function, refresh: Function,
 *   dispose: Function}} The band, in the shape a report panel's sketch takes.
 */
function createFalseColorBand(node) {
  const element = document.createElement("div");
  element.style.cssText = "display:flex;flex-direction:column;gap:6px;flex:1 1 auto;min-height:0;"
    + "min-width:0";

  const picture = createPictureBand(node, {
    slot: FALSE_COLOR_SLOT,
    label: "ARRI false colour",
    height: FALSE_COLOR_PICTURE_HEIGHT,
  });
  element.appendChild(picture.element);

  const legend = document.createElement("div");
  legend.style.cssText = "flex:0 0 auto;display:grid;align-items:center;column-gap:8px;row-gap:3px;"
    + "grid-template-columns:auto fit-content(50%) minmax(0,1fr) auto;padding:6px 8px;"
    + `border-radius:6px;background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};`
    + "font-variant-numeric:tabular-nums";
  element.appendChild(legend);

  /**
   * Add one legend row.
   *
   * @param {string|null} colour - A hex code for the swatch, or null for the grey rest.
   * @param {string} name - What the band is.
   * @param {string} range - Its signal levels in words.
   * @param {number} share - Percent of the frame in it.
   * @returns {void}
   */
  const row = (colour, name, range, share) => {
    const swatch = document.createElement("span");
    swatch.style.cssText = "width:12px;height:12px;border-radius:3px;"
      + `border:1px solid ${CARD_LOOK.edge}`;
    if (colour) swatch.style.backgroundColor = colour;
    else swatch.style.backgroundImage = "linear-gradient(90deg,#000,#fff)";
    const label = document.createElement("span");
    label.style.cssText = `color:${CARD_LOOK.text};white-space:nowrap;overflow:hidden;`
      + "text-overflow:ellipsis;min-width:0";
    label.textContent = name;
    const levels = document.createElement("span");
    levels.style.cssText = `color:${themeVar("fgMuted")};white-space:nowrap;overflow:hidden;`
      + "text-overflow:ellipsis;min-width:0";
    levels.textContent = range;
    const figure = document.createElement("span");
    figure.style.cssText = `text-align:right;white-space:nowrap;font-weight:650;color:${
      share > 0 ? CARD_LOOK.title : themeVar("fgMuted")}`;
    figure.textContent = `${Number(share).toFixed(2)}%`;
    legend.append(swatch, label, levels, figure);
  };

  /**
   * Draw the legend from a report.
   *
   * @param {object|null} report - The false colour report, or null before a run.
   * @returns {void}
   */
  const drawLegend = (report) => {
    legend.textContent = "";
    const tallies = Array.isArray(report?.tallies) ? report.tallies : [];
    if (!tallies.length) {
      legend.style.display = "none";
      return;
    }
    legend.style.display = "grid";
    const colours = factOf(report, COLOURS_FACT).split(/\s+/);
    const levels = factOf(report, LEVELS_FACT).split(/\s+/);
    const bands = Math.min(tallies.length, colours.length);
    tallies.forEach((tally, index) => {
      const share = Number(tally?.value) || 0;
      if (index < bands) {
        const colour = HEX_COLOUR.test(colours[index]) ? colours[index] : null;
        row(colour, String(tally?.name ?? ""), rangeText(levels[index], index, bands), share);
      } else {
        row(null, String(tally?.name ?? ""), "", share);
      }
    });
  };

  drawLegend(null);

  return {
    element,
    update(report) {
      picture.update(report);
      drawLegend(report);
    },
    clear() {
      picture.clear();
      drawLegend(null);
    },
    refresh: () => picture.refresh(),
    dispose() {
      picture.dispose();
    },
  };
}

/**
 * Build the panel one scope node draws.
 *
 * @param {object} node - The node the panel is on.
 * @param {string} type - The node's id.
 * @returns {object} The report panel, for `appendInterfaceWidget`.
 */
function createScopePanel(node, type) {
  if (type === FALSE_COLOR) {
    return createReportPanel(node, {
      className: "was-image-false-color",
      facts: false,
      height: FALSE_COLOR_PANEL_HEIGHT,
      minWidth: FALSE_COLOR_MIN_WIDTH,
      emptyLabel: "No bands measured yet",
      logName: EXT_NAME,
      failure: "Failed to read the false colour report:",
      sketch: () => createFalseColorBand(node),
    });
  }
  return createReportPanel(node, {
    className: "was-image-vectorscope",
    height: SCOPE_PANEL_HEIGHT,
    minWidth: PANEL_MIN_WIDTH,
    emptyLabel: "No colours plotted yet",
    logName: EXT_NAME,
    failure: "Failed to read the vectorscope report:",
    sketch: () => createPictureBand(node, {
      slot: SCOPE_SLOT,
      label: "Rec.709 Cb and Cr",
      height: SCOPE_PICTURE_HEIGHT,
    }),
  });
}

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_IDS.has(nodeData?.name)) return;
    const type = nodeData.name;

    const proto = nodeType.prototype;
    if (proto.__was_image_scopes_wrapped) return;
    proto.__was_image_scopes_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createScopePanel(this, type);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the scope panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the scope panel:`, error);
      }
      return result;
    };
  },
});
