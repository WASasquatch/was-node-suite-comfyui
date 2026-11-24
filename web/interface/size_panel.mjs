/**
 * What a geometry node did to a frame's size, drawn on the node itself.
 *
 * The band holds two nested rectangles at one scale, the frame that went in as an outline and
 * the frame that came out filled.
 */

import { createReportPanel } from "./report_panel.mjs";
import { themeVar } from "./theme.mjs";

// Height in node units the panel opens at, which shows the summary, the four figures, the six
// rows and the band without scrolling. The node's spare room is taken on top of it.
export const PANEL_HEIGHT = 208;

// Height in node units of the summary-only panel, for a node that already carries an editor and
// has one fact to add to it.
export const SHORT_HEIGHT = 36;

// The narrowest the panel can be drawn in, in node units: the four figures across the top, and
// below them the sketch, the gap and the widest size label the family writes.
const MIN_WIDTH = 260;

// The box the two rectangles are fitted inside, in pixels. Both are scaled by one factor, so a
// 4:3 frame and a 1:1 frame keep their own shapes and can be compared by eye.
const SKETCH_WIDTH = 132;
const SKETCH_HEIGHT = 56;

// What a size fact opens with. `size_report.py` writes the unit after the pair, so a latent's
// `64x96 latent` reads the same two numbers as a picture's `64x96`.
const SIZE_PATTERN = /^\s*(\d+)\s*x\s*(\d+)/;

// The fact-name column, in CSS pixels. Wide enough for `megapixels` at 11px monospace, which is
// the widest name the size report writes.
const LABEL_WIDTH = 76;

// The words drawn before a node has run.
const EMPTY_LABEL = "No size to report yet";

// How much of the status colour the filled rectangle and its swatch carry.
const FILL_ALPHA = "26.7%";

const LOG_NAME = "WASNodeSuite.SizePanel";

/**
 * One named value out of a report.
 *
 * @param {object} report - The report, whose `facts` is an array of `{name, value}`.
 * @param {string} name - The fact to find.
 * @returns {string} The value, or an empty string when the report carries no such fact.
 */
function fact(report, name) {
  for (const row of report?.facts ?? []) {
    if (row?.name === name) return String(row.value ?? "");
  }
  return "";
}

/**
 * A size fact as two numbers.
 *
 * @param {string} said - The fact's value, such as `1024x1536` or `64x96 latent`.
 * @returns {{width: number, height: number}|null} The pair, or null when it is not one.
 */
function readSize(said) {
  const found = SIZE_PATTERN.exec(said);
  if (!found) return null;
  const width = Number(found[1]);
  const height = Number(found[2]);
  return width > 0 && height > 0 ? { width, height } : null;
}

/**
 * The two rectangles in pixels, at one scale that fits the larger of them in the box.
 *
 * @param {{width: number, height: number}} before - The frame that went in.
 * @param {{width: number, height: number}} after - The frame that came out.
 * @returns {{stage: number[], before: number[], after: number[]}} Each as `[width, height]`.
 */
function fit(before, after) {
  const widest = Math.max(before.width, after.width);
  const tallest = Math.max(before.height, after.height);
  const scale = Math.min(SKETCH_WIDTH / widest, SKETCH_HEIGHT / tallest);
  const box = (size) => [
    Math.max(1, Math.round(size.width * scale)),
    Math.max(1, Math.round(size.height * scale)),
  ];
  return {
    stage: [Math.max(1, Math.round(widest * scale)), Math.max(1, Math.round(tallest * scale))],
    before: box(before),
    after: box(after),
  };
}

/**
 * Build the proportion sketch, as the band `report_panel.mjs` draws under the figures.
 *
 * @returns {{element: HTMLElement, update: (report: object) => void, clear: () => void,
 *   dispose: () => void}} The band, in the shape `createReportPanel` calls a sketch in.
 */
export function createSizeSketch() {
  const band = document.createElement("div");
  band.className = "was-size-sketch";
  band.style.cssText =
    "display:flex;align-items:center;gap:12px;flex:0 0 auto;min-height:0;overflow:hidden";

  const stage = document.createElement("div");
  stage.style.cssText = "position:relative;flex:0 0 auto";
  band.appendChild(stage);

  const inputBox = document.createElement("div");
  const outputBox = document.createElement("div");
  for (const box of [inputBox, outputBox]) {
    box.style.cssText =
      "position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);box-sizing:border-box";
    stage.appendChild(box);
  }

  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-direction:column;gap:3px;min-width:0;flex:1 1 auto";
  band.appendChild(legend);

  /**
   * One legend row: a swatch in the rectangle's own colour, its side, and its size.
   *
   * @param {string} side - `in` or `out`.
   * @returns {{row: HTMLElement, swatch: HTMLElement, text: HTMLElement}} The row and its parts.
   */
  const buildRow = (side) => {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:6px;min-width:0";
    const swatch = document.createElement("span");
    swatch.style.cssText = "flex:0 0 auto;width:8px;height:8px;box-sizing:border-box";
    const text = document.createElement("span");
    text.style.cssText =
      "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
    text.dataset.side = side;
    row.append(swatch, text);
    legend.appendChild(row);
    return { row, swatch, text };
  };

  const inputRow = buildRow("in");
  const outputRow = buildRow("out");

  /**
   * Redraw the band from one report.
   *
   * @param {object} answer - A run_result envelope or the report inside it.
   * @returns {void} Nothing. The band is hidden where the report carries no two sizes.
   */
  const update = (answer) => {
    const report = answer && typeof answer === "object" && "result" in answer
      ? answer.result
      : answer;
    const before = readSize(fact(report, "in"));
    const after = readSize(fact(report, "out"));
    if (!before || !after) {
      band.style.display = "none";
      return;
    }
    band.style.display = "flex";

    // The output carries the status colour, since the output is what the status is about.
    const filled = themeVar(report?.status === "error"
      ? "error"
      : report?.status === "warning" ? "warning" : "accent");
    const wash = `color-mix(in srgb, ${filled} ${FILL_ALPHA}, transparent)`;
    const boxes = fit(before, after);

    stage.style.width = `${boxes.stage[0]}px`;
    stage.style.height = `${boxes.stage[1]}px`;

    inputBox.style.width = `${boxes.before[0]}px`;
    inputBox.style.height = `${boxes.before[1]}px`;
    inputBox.style.border = `1px solid ${themeVar("fgMuted")}`;
    inputBox.style.background = "transparent";

    outputBox.style.width = `${boxes.after[0]}px`;
    outputBox.style.height = `${boxes.after[1]}px`;
    outputBox.style.border = `1px solid ${filled}`;
    outputBox.style.background = wash;

    inputRow.swatch.style.border = `1px solid ${themeVar("fgMuted")}`;
    inputRow.swatch.style.background = "transparent";
    inputRow.text.style.color = themeVar("fgMuted");
    inputRow.text.textContent = `in  ${fact(report, "in")}`;

    outputRow.swatch.style.border = `1px solid ${filled}`;
    outputRow.swatch.style.background = wash;
    outputRow.text.style.color = themeVar("fg");
    outputRow.text.textContent = `out ${fact(report, "out")}`;
  };

  return {
    element: band,
    update,
    clear: () => {
      band.style.display = "none";
    },
    dispose() {
      band.replaceChildren();
    },
  };
}

/**
 * Build the panel one geometry node draws its size report in.
 *
 * @param {object} node - The node the panel belongs to, for its id and its redraws.
 * @param {{height?: number, sketch?: boolean|Function, tiles?: boolean, facts?: boolean,
 *   grows?: boolean}} [options] - What the node wants drawn. A node already carrying an
 *   editor turns the figures, the rows and the sketch off and keeps the summary. A function
 *   given as `sketch` is the band drawn in place of the two rectangles.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, minWidth: number,
 *   update: Function, clear: Function, refresh: Function, dispose: Function}} The panel, for
 *   `appendInterfaceWidget`.
 */
export function createSizePanel(node, options = {}) {
  const height = Number(options.height) > 0 ? Number(options.height) : PANEL_HEIGHT;
  const given = typeof options.sketch === "function" ? options.sketch : null;
  const wantsSketch = options.sketch !== false;

  const panel = createReportPanel(node, {
    summary: true,
    tiles: options.tiles !== false,
    facts: options.facts !== false,
    // The sketch is a factory the panel calls once and then owns: it appends the band between
    // the counts and the fact rows, hands it the report on every draw, and releases it.
    sketch: given ?? (wantsSketch ? createSizeSketch : null),
    height,
    // `megapixels` is the widest name the family writes, and it elides at the column a report
    // panel opens with.
    labelWidth: LABEL_WIDTH,
    emptyLabel: EMPTY_LABEL,
    className: "was-size-report",
    logName: LOG_NAME,
    failure: "Failed to read the size report:",
  });

  return {
    element: panel.element,
    height,
    // A node told not to grow leaves every spare unit to the interface it already carries.
    maxHeight: options.grows === false ? height : Number.MAX_SAFE_INTEGER,
    minWidth: wantsSketch ? MIN_WIDTH : 0,
    update: panel.update,
    clear: panel.clear,
    refresh: panel.refresh,
    dispose: panel.dispose,
  };
}
