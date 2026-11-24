/**
 * What a mask operation did to a mask, drawn on the node that did it.
 *
 * The band draws the mask's bounding box inside a rectangle at the mask's own aspect, both as
 * percentages.
 */

import { createReportPanel } from "./report_panel.mjs";
import { themeVar } from "./theme.mjs";

const LOG_NAME = "WASNodeSuite.MaskState";

// Height in node units the panel opens at, the frame band included. Enough for the summary, the
// figure tiles, the band and six fact rows without scrolling.
const PANEL_HEIGHT = 188;

// Height in node units the frame band is given. Enough for a box near the top and a box near
// the bottom to be told apart, and small enough that the numbers keep the node.
const BAND_HEIGHT = 56;

// Narrowest the node is worth drawing at, so the fact values are not all ellipsis. Answered
// from `computeLayoutSize`, which is what the frontend counts when it refits a node.
const MIN_WIDTH = 240;

// `WxH`, and the size the mask arrived at where the node changed it.
const SIZE = /^(\d+)x(\d+)(?:,\s*from\s*(\d+)x(\d+))?$/;

// `x<left> y<top> <width>x<height>`, as `modules/interface/mask_report.py` writes a rectangle.
const BOX = /^x(-?\d+)\s+y(-?\d+)\s+(\d+)x(\d+)/;

/**
 * Build the measurement band one mask node draws its report in.
 *
 * @param {object} node - The node the band belongs to, for its id and its redraws.
 * @returns {object} The panel `createReportPanel` answers, for `appendInterfaceWidget`.
 */
export function createMaskStatePanel(node) {
  return createReportPanel(node, {
    summary: true,
    tiles: true,
    facts: true,
    sketch: createFrameBand,
    height: PANEL_HEIGHT,
    minWidth: MIN_WIDTH,
    emptyLabel: "No mask measured yet",
    className: "was-mask-state",
    logName: LOG_NAME,
    failure: "Failed to read the mask report:",
  });
}

/**
 * Build the frame the mask's bounding box is drawn inside.
 *
 * @returns {{element: HTMLElement, update: (report: object) => void, clear: () => void,
 *   dispose: () => void}} The band, in the shape `createReportPanel` calls a sketch in.
 */
export function createFrameBand() {
  const stage = document.createElement("div");
  stage.style.cssText = "flex:0 0 auto;display:flex;align-items:center;justify-content:center;"
    + `height:${BAND_HEIGHT}px;overflow:hidden`;

  // Two nested rectangles rather than a canvas: a canvas built once is a bitmap the graph
  // magnifies, and one rebuilt from the zoom needs a ratio watcher for a picture that is two
  // rectangles.
  const frame = document.createElement("div");
  frame.style.cssText = "position:relative;box-sizing:border-box;"
    + `border:1px solid ${themeVar("border")};background:${themeVar("bgDark")}`;
  const box = document.createElement("div");
  box.style.cssText = "position:absolute;box-sizing:border-box;"
    + `border:1px solid ${themeVar("accent")};background:${themeVar("accentBg")}`;
  frame.appendChild(box);
  stage.appendChild(frame);

  /**
   * Fit the frame and its box to a report, or draw neither where there is nothing to fit.
   *
   * @param {object|null} report - The report the panel is drawing.
   * @returns {void}
   */
  const update = (report) => {
    const facts = new Map((report?.facts ?? []).map(({ name, value }) => [name, value]));
    const size = SIZE.exec(facts.get("size") ?? "");
    const width = size ? Number(size[1]) : 0;
    const height = size ? Number(size[2]) : 0;
    if (!(width > 0) || !(height > 0)) {
      stage.style.display = "none";
      return;
    }
    stage.style.display = "flex";
    // The frame keeps the mask's aspect inside the band's own height, so a wide mask and a
    // tall one are told apart before either rectangle is read.
    const scale = Math.min(1, BAND_HEIGHT / height);
    frame.style.height = `${Math.max(1, Math.round(height * scale))}px`;
    frame.style.width = `${Math.max(1, Math.round(width * scale))}px`;
    frame.title = `${width} by ${height}, with the box the mask fills inside it`;

    const marked = BOX.exec(facts.get("box") ?? "");
    if (!marked) {
      box.style.display = "none";
      return;
    }
    box.style.display = "block";
    box.style.left = `${percent(Number(marked[1]), width)}%`;
    box.style.top = `${percent(Number(marked[2]), height)}%`;
    // A box one pixel wide on a 4096 mask rounds to nothing, so both sides keep a floor the
    // rectangle is still visible at.
    box.style.width = `${Math.max(2, percent(Number(marked[3]), width))}%`;
    box.style.height = `${Math.max(2, percent(Number(marked[4]), height))}%`;
  };

  return {
    element: stage,
    update,
    clear: () => {
      stage.style.display = "none";
    },
    dispose: () => {},
  };
}

/**
 * One length as a percentage of another, held inside the frame.
 *
 * @param {number} part - The length, in pixels of the mask.
 * @param {number} whole - The frame's own length in the same axis.
 * @returns {number} The share, between 0 and 100, to two decimals.
 */
function percent(part, whole) {
  if (!Number.isFinite(part) || !(whole > 0)) return 0;
  return Math.round(Math.min(100, Math.max(0, (100 * part) / whole)) * 100) / 100;
}
