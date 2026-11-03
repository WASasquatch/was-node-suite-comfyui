/**
 * The health of a training set, drawn on Dataset Health.
 *
 * A radar of the published scores, the table behind them, the flag counts, shared phrases, the
 * flagged pictures and near-duplicate pairs with thumbnails, each key explained on hover.
 */

import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { CARD_LOOK, cardChip } from "./interface/card.js";
import { createReportPanel } from "./interface/report_panel.js";
import { fetchRunResultPage } from "./interface/run_result.js";
import { themeVar } from "./interface/theme.js";
import { appendInterfaceWidget } from "./interface/widget.js";
import { LOOK, SANS } from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.DatasetHealthUI";
const NODE_ID = "WASDatasetHealth";

const UI_WIDGET_NAME = "was_dataset_health_ui";
const UI_WIDGET_TYPE = "was_dataset_health";

// Height in node units the panel opens at: the summary, the radar and its table, the flag
// counts and the first rows of the flagged pictures.
const PANEL_HEIGHT = 460;

// Width in node units a new node opens at.
const PANEL_WIDTH = 270;

const EMPTY_LABEL = "run the node to measure the set";

// The scores in the order the radar draws them, clockwise from the top.
const SCORES = [
  ["captions", "Captions"],
  ["consistency", "Consistency"],
  ["quality", "Quality"],
  ["uniqueness", "Uniqueness"],
  ["balance", "Balance"],
  ["faces", "Faces"],
];

// The rows of the table, each named by the fact the node published for it.
const ROWS = [
  ["captions", "Captions"],
  ["consistency", "Consistency"],
  ["quality", "Quality"],
  ["exposure", "Exposure"],
  ["uniqueness", "Uniqueness"],
  ["balance", "Balance"],
  ["faces", "Faces"],
];

// What each score and table row measures, shown on hover over its name and value.
const HINTS = {
  captions: "Share of pictures with a caption",
  consistency: "Even caption lengths; repeats and short captions lower it",
  quality: "Share with no picture-quality flag",
  exposure: "Median brightness; spread from the 10th to the 90th percentile",
  uniqueness: "Share with no near-duplicate",
  balance: "How evenly pictures spread over aspect ratios; one ratio alone scores 100",
  faces: "Share with exactly one main face",
};

// What earns each flag, shown on hover over its chip.
const FLAG_HINTS = {
  unreadable: "Could not be read as a picture",
  dark: "Brightness under 15%",
  "blown out": "Brightness over 85%",
  flat: "Low contrast: brightness deviation under 6%",
  blurry: "Sharpness under a quarter of the set's median",
  bars: "Flat bars along the edges over 8% of the picture",
  greyscale: "No colour in a colour set",
  transparent: "Over 1% of the picture see-through",
  compressed: "JPEG quality under 60",
  small: "Short side under target_side",
  "near-duplicate": "A resized, re-saved or mirrored copy of another",
  "no face": "No face found",
  "several faces": "A second face at least half the main one's height",
  "small face": "Main face under 5% of the picture's height",
};

// The axes drawn before the first run.
const EMPTY_AXES = 5;

// The radar's geometry in its own units: the outer ring's radius, where a label sits beyond
// it, and the box the labels fit in.
const RADIUS = 54;
const LABEL_RADIUS = 66;
const VIEW_BOX = "-122 -92 244 186";
const SVG_NS = "http://www.w3.org/2000/svg";

// The widest and narrowest the radar is drawn, in CSS pixels.
const RADAR_MAX = 250;
const RADAR_MIN = 140;

// Rows of each list drawn at most, read a page at a time past what the report carried.
const MAX_FLAGGED = 120;
const MAX_PAIRS = 60;

// Lines of a dataset run's thumbnail labels read: a flagged picture and a pair's two at a time.
const MAX_THUMB_LINES = 3 * Math.max(MAX_FLAGGED, MAX_PAIRS);

// Phrase chips and flag chips drawn at most.
const MAX_PHRASES = 12;
const MAX_FLAG_KINDS = 16;

// A thumbnail's side as drawn, and the edge it is asked for, in CSS pixels.
const THUMB_SIZE = 40;
const THUMB_EDGE = 96;
const THUMB_ROUTE = "/was/interface/api/file_thumbnail";

// Scores at or above these read as good and as fair; anything lower is a warning.
const GOOD = 0.75;
const FAIR = 0.5;

/**
 * The colour a score is drawn in.
 *
 * @param {number} score - A score from 0 to 1, or anything else for none.
 * @returns {string} A `var()` reference to a palette property.
 */
function scoreColour(score) {
  if (!Number.isFinite(score)) return LOOK.muted;
  if (score >= GOOD) return LOOK.success;
  if (score >= FAIR) return LOOK.text;
  return LOOK.warning;
}

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
 * An SVG element with attributes.
 *
 * @param {string} tag - The tag.
 * @param {object} attributes - Attribute names to values.
 * @returns {SVGElement} The element.
 */
function svg(tag, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
  return element;
}

/**
 * A small uppercase heading over one block of the panel.
 *
 * @param {string} text - What it says.
 * @returns {HTMLElement} The heading.
 */
function heading(text) {
  return el(
    "div",
    `flex:0 0 auto;font:600 9px ${SANS};letter-spacing:0.08em;text-transform:uppercase;`
      + `color:${themeVar("fgMuted")};margin-top:2px`,
    text,
  );
}

/**
 * The listing label of one picture of the measured folder.
 *
 * @param {string} folder - The folder as its menu names it, `set [input]` or a bare root.
 * @param {string} name - The picture's path below the folder.
 * @returns {string} The label, or empty where the folder is not named.
 */
function fileLabel(folder, name) {
  const found = /^(.*) \[([^\]]+)\]$/.exec(folder);
  if (found) return `${found[1]}/${name} [${found[2]}]`;
  return folder ? `${name} [${folder}]` : "";
}

/**
 * One line of a body split at its last `: `, into what it names and what it says.
 *
 * @param {string} line - The line.
 * @returns {[string, string]} The two halves; the second is empty where there is no `: `.
 */
function splitLine(line) {
  const cut = line.lastIndexOf(": ");
  return cut < 0 ? [line, ""] : [line.slice(0, cut), line.slice(cut + 2)];
}

/**
 * Lines one body of a report holds, a page read past what the report carried.
 *
 * @param {object} node - The node the report belongs to.
 * @param {object} report - The report.
 * @param {string} name - The body's name.
 * @param {number} wanted - How many lines to answer at most.
 * @returns {Promise<{lines: string[], total: number}>} The lines and how many the body holds.
 */
async function bodyLines(node, report, name, wanted) {
  const bodies = Array.isArray(report?.bodies) ? report.bodies : [];
  const index = bodies.findIndex((body) => body?.name === name);
  if (index < 0) return { lines: [], total: 0 };
  const body = bodies[index];
  let lines = body.text ? body.text.split("\n") : [];
  // A piece cut to a character count may end part way through a line.
  if (body.whole === false && lines.length) lines.pop();
  const total = Math.max(Number(body.lines) || 0, lines.length);
  if (lines.length < Math.min(wanted, total)) {
    const answer = await fetchRunResultPage(node, index, 0, wanted);
    const page = answer?.page;
    if (page && page.name === name && (!page.run || page.run === report.run)) lines = page.lines;
  }
  return { lines: lines.filter((line) => line).slice(0, wanted), total };
}

/**
 * One picture's thumbnail, drawn as an empty tile where it cannot be read.
 *
 * @param {string} label - The picture's listing label.
 * @param {number} run - The run the report was published on, so a new run asks again.
 * @returns {HTMLElement} The tile.
 */
function thumbnail(label, run) {
  const tile = el(
    "div",
    `flex:0 0 auto;width:${THUMB_SIZE}px;height:${THUMB_SIZE}px;border-radius:4px;overflow:hidden;`
      + `background:${CARD_LOOK.inset};border:1px solid ${LOOK.border};box-sizing:border-box`,
  );
  if (!label) return tile;
  const image = document.createElement("img");
  image.alt = "";
  image.loading = "lazy";
  image.decoding = "async";
  image.draggable = false;
  image.style.cssText = "display:block;width:100%;height:100%;object-fit:cover;pointer-events:none";
  image.addEventListener("error", () => image.remove(), { once: true });
  image.src = api.apiURL(
    `${THUMB_ROUTE}?label=${encodeURIComponent(label)}&edge=${THUMB_EDGE}`
      + `&stamp=${encodeURIComponent(String(run ?? ""))}`,
  );
  tile.appendChild(image);
  return tile;
}

/**
 * Build the radar, the table and the lists a report panel draws between its summary and rows.
 *
 * @param {object} node - The node the panel belongs to, for its report's pages.
 * @returns {{element: HTMLElement, update: (report: object) => void, clear: () => void,
 *   dispose: () => void}} The sketch, in the shape `createReportPanel` takes.
 */
function createHealthSketch(node) {
  const element = el(
    "div",
    "flex:1 1 auto;min-height:0;display:flex;flex-direction:column;gap:6px;"
      + "font-variant-numeric:tabular-nums",
  );

  const top = el(
    "div",
    "flex:0 0 auto;display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;"
      + "justify-content:center",
  );
  const radarBox = el(
    "div",
    `flex:1 1 190px;min-width:${RADAR_MIN}px;max-width:${RADAR_MAX}px;line-height:0`,
  );
  const radar = svg("svg", { viewBox: VIEW_BOX, role: "img" });
  radar.style.cssText = "display:block;width:100%;height:auto;overflow:visible";
  radarBox.appendChild(radar);

  const table = el(
    "div",
    "flex:1 1 170px;min-width:0;display:grid;"
      + "grid-template-columns:fit-content(50%) minmax(0,1fr) auto;column-gap:8px;row-gap:2px;"
      + "align-items:baseline",
  );
  top.append(radarBox, table);

  const flagBlock = el("div", "flex:0 0 auto;display:none;flex-direction:column;gap:3px");
  const flagChips = el("div", "display:flex;flex-wrap:wrap;gap:4px");
  flagBlock.append(heading("Flags"), flagChips);

  const phraseBlock = el("div", "flex:0 0 auto;display:none;flex-direction:column;gap:3px");
  const phraseChips = el("div", "display:flex;flex-wrap:wrap;gap:4px");
  phraseBlock.title = "Share of captions holding each";
  phraseBlock.append(heading("Shared phrases"), phraseChips);

  const lists = el(
    "div",
    "flex:1 1 auto;min-height:0;overflow:auto;display:none;flex-direction:column;gap:4px;"
      + `scrollbar-width:thin;scrollbar-color:${LOOK.muted} transparent`,
  );

  element.append(top, flagBlock, phraseBlock, lists);

  let disposed = false;
  let drawnRun = null;
  let serial = 0;

  /**
   * Draw the radar for a set of scores, or its empty frame.
   *
   * @param {Array<[string, string, number|null]>} axes - Each axis's key, name and score, null
   *   for none.
   * @returns {void}
   */
  const drawRadar = (axes) => {
    radar.textContent = "";
    const count = axes.length;
    const corner = (index, radius) => {
      const angle = -Math.PI / 2 + (index * 2 * Math.PI) / count;
      return [Math.cos(angle) * radius, Math.sin(angle) * radius];
    };
    const ring = (share) => axes.map((_, index) => corner(index, RADIUS * share).join(",")).join(" ");
    for (const share of [0.25, 0.5, 0.75, 1]) {
      const outline = svg("polygon", { points: ring(share), fill: "none" });
      outline.style.stroke = themeVar("border");
      outline.style.strokeWidth = share === 1 ? "1" : "0.6";
      radar.appendChild(outline);
    }
    axes.forEach((_, index) => {
      const [x, y] = corner(index, RADIUS);
      const spoke = svg("line", { x1: 0, y1: 0, x2: x.toFixed(2), y2: y.toFixed(2) });
      spoke.style.stroke = themeVar("border");
      spoke.style.strokeWidth = "0.6";
      radar.appendChild(spoke);
    });

    const scored = axes.every(([, , score]) => Number.isFinite(score));
    if (scored) {
      const points = axes.map(([, , score], index) => corner(index, RADIUS * Math.max(0, Math.min(1, score))));
      const shape = svg("polygon", { points: points.map((point) => point.join(",")).join(" ") });
      shape.style.fill = `color-mix(in srgb, ${CARD_LOOK.title} 24%, transparent)`;
      shape.style.stroke = CARD_LOOK.title;
      shape.style.strokeWidth = "1.4";
      shape.style.strokeLinejoin = "round";
      radar.appendChild(shape);
      points.forEach(([x, y], index) => {
        const dot = svg("circle", { cx: x.toFixed(2), cy: y.toFixed(2), r: 2.4 });
        dot.style.fill = scoreColour(axes[index][2]);
        radar.appendChild(dot);
      });
    }

    axes.forEach(([key, name, score], index) => {
      const [x, y] = corner(index, LABEL_RADIUS);
      const anchor = Math.abs(x) < 6 ? "middle" : x > 0 ? "start" : "end";
      // Labels above the centre sit with their value on top; below it, under the name.
      const above = y < -6;
      const label = svg("text", { x: x.toFixed(2), "text-anchor": anchor });
      label.style.font = `9px ${SANS}`;
      const hint = svg("title");
      hint.textContent = HINTS[key] ?? name;
      const title = svg("tspan", { x: x.toFixed(2), y: (y + (above ? -2 : 3)).toFixed(2) });
      title.style.fill = themeVar("fgMuted");
      title.textContent = name;
      const value = svg("tspan", { x: x.toFixed(2), y: (y + (above ? -12 : 14)).toFixed(2) });
      value.style.fill = scoreColour(score);
      value.style.fontWeight = "700";
      value.style.fontSize = "11px";
      value.textContent = Number.isFinite(score) ? String(Math.round(score * 100)) : "-";
      label.append(hint, title, value);
      radar.appendChild(label);
    });
    radar.setAttribute(
      "aria-label",
      axes.map(([, name, score]) => `${name} ${Number.isFinite(score) ? Math.round(score * 100) : "none"}`).join(", "),
    );
  };

  /**
   * Draw the table of what each score measured.
   *
   * @param {object|null} report - The report, or null for the table's empty rows.
   * @returns {void}
   */
  const drawTable = (report) => {
    table.textContent = "";
    const facts = new Map((report?.facts ?? []).map((entry) => [entry.name, entry.value]));
    const scores = new Map((report?.tallies ?? []).map((entry) => [entry.name, entry.value]));
    for (const [key, name] of ROWS) {
      if (report && !facts.has(key)) continue;
      if (!report && key === "faces") continue;
      const said = String(facts.get(key) ?? "");
      const cut = said.indexOf("; ");
      const value = cut < 0 ? said : said.slice(0, cut);
      const detail = cut < 0 ? "" : said.slice(cut + 2);
      const score = scores.has(key) ? Number(scores.get(key)) : NaN;
      const row = el("div", "display:contents");
      row.title = detail ? `${name}: ${detail}` : "";
      const label = el("span", `color:${themeVar("fgMuted")};white-space:nowrap;overflow:hidden;`
        + "text-overflow:ellipsis;min-width:52px", name);
      const measured = el("span", `color:${CARD_LOOK.text};white-space:nowrap;overflow:hidden;`
        + "text-overflow:ellipsis;min-width:0", value || "-");
      const figure = el(
        "span",
        `text-align:right;font-weight:700;color:${scoreColour(score)}`,
        Number.isFinite(score) ? String(Math.round(score * 100)) : "",
      );
      for (const cell of [measured, figure]) cell.title = row.title;
      label.title = HINTS[key] ?? row.title;
      row.append(label, measured, figure);
      table.appendChild(row);
    }
  };

  /**
   * Draw the flag counts and the phrase chips.
   *
   * @param {object} report - The report.
   * @returns {void}
   */
  const drawChips = (report) => {
    const bodies = new Map((report.bodies ?? []).map((body) => [body.name, body.text || ""]));
    flagChips.textContent = "";
    for (const line of (bodies.get("flags") ?? "").split("\n").slice(0, MAX_FLAG_KINDS)) {
      const found = /^(\d+) (.+)$/.exec(line);
      if (!found) continue;
      const chip = cardChip(`${found[1]} ${found[2]}`, "warning");
      chip.title = FLAG_HINTS[found[2]] ?? "";
      flagChips.appendChild(chip);
    }
    flagBlock.style.display = flagChips.childElementCount ? "flex" : "none";

    phraseChips.textContent = "";
    const phrases = [
      ...(bodies.get("frequent phrases") ?? "").split("\n").map((line) => [line, "warning"]),
      ...(bodies.get("phrases") ?? "").split("\n").map((line) => [line, "plain"]),
    ];
    for (const [line, tone] of phrases) {
      if (phraseChips.childElementCount >= MAX_PHRASES) break;
      const found = /^(\d+%) (.+)$/.exec(line);
      if (!found) continue;
      const chip = cardChip(`${found[2]} ${found[1]}`, tone);
      chip.style.textTransform = "none";
      phraseChips.appendChild(chip);
    }
    phraseBlock.style.display = phraseChips.childElementCount ? "flex" : "none";
  };

  /**
   * A grid the rows of one list run across in as many columns as the panel holds.
   *
   * @returns {HTMLElement} The grid.
   */
  const rowGrid = () => el(
    "div",
    "flex:0 0 auto;display:grid;gap:4px;"
      + "grid-template-columns:repeat(auto-fill,minmax(min(100%,210px),1fr))",
  );

  /**
   * One row of a list: thumbnails, then lines of text.
   *
   * @param {HTMLElement[]} tiles - The thumbnails.
   * @param {Array<[string, string]>} lines - Each line's text and colour.
   * @param {string} hint - What the row says on hover.
   * @returns {HTMLElement} The row.
   */
  const listRow = (tiles, lines, hint) => {
    const row = el(
      "div",
      "display:flex;gap:6px;align-items:center;min-width:0;padding:3px;border-radius:6px;"
        + `background:${CARD_LOOK.inset}`,
    );
    row.title = hint;
    const words = el("div", "flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:1px");
    for (const [text, colour] of lines) {
      words.appendChild(el(
        "span",
        `color:${colour};white-space:nowrap;overflow:hidden;text-overflow:ellipsis`,
        text,
      ));
    }
    row.append(...tiles, words);
    return row;
  };

  /**
   * A muted line saying how many rows were left out.
   *
   * @param {number} more - How many.
   * @returns {HTMLElement} The line.
   */
  const moreLine = (more) => el("div", `flex:0 0 auto;color:${themeVar("fgMuted")};font-size:10px`,
    `${more} more in the report`);

  /**
   * Draw the flagged pictures and the near-duplicate pairs.
   *
   * @param {object} report - The report.
   * @returns {Promise<void>}
   */
  const drawLists = async (report) => {
    const asked = ++serial;
    const folder = String((report.facts ?? []).find((entry) => entry.name === "folder")?.value ?? "");
    const [flagged, details, pairs, thumbs] = await Promise.all([
      bodyLines(node, report, "flagged", MAX_FLAGGED),
      bodyLines(node, report, "details", MAX_FLAGGED),
      bodyLines(node, report, "near-duplicates", MAX_PAIRS),
      bodyLines(node, report, "thumbnails", MAX_THUMB_LINES),
    ]);
    if (disposed || asked !== serial) return;
    const scrolled = lists.scrollTop;
    lists.textContent = "";
    const described = new Map(details.lines.map((line) => splitLine(line)));
    // A dataset run names each picture's label; a folder run's labels follow from the folder.
    const labels = new Map(thumbs.lines.map((line) => {
      const cut = line.indexOf("\t");
      return cut < 0 ? [line, ""] : [line.slice(0, cut), line.slice(cut + 1)];
    }));
    const labelOf = (name) => labels.get(name) ?? fileLabel(folder, name);

    if (flagged.lines.length) {
      lists.appendChild(heading(`Flagged pictures (${flagged.total})`));
      const grid = rowGrid();
      for (const line of flagged.lines) {
        const [name, reasons] = splitLine(line);
        const said = described.get(name);
        grid.appendChild(listRow(
          [thumbnail(labelOf(name), report.run)],
          [[name, CARD_LOOK.text], [reasons, LOOK.warning]],
          said ? `${name}: ${said}` : line,
        ));
      }
      lists.appendChild(grid);
      if (flagged.total > flagged.lines.length) lists.appendChild(moreLine(flagged.total - flagged.lines.length));
    }

    if (pairs.lines.length) {
      lists.appendChild(heading(`Near-duplicates (${pairs.total})`));
      const grid = rowGrid();
      for (const line of pairs.lines) {
        const [both, detail] = splitLine(line);
        const cut = both.indexOf(" | ");
        const first = cut < 0 ? both : both.slice(0, cut);
        const second = cut < 0 ? "" : both.slice(cut + 3);
        grid.appendChild(listRow(
          [thumbnail(labelOf(first), report.run), thumbnail(labelOf(second), report.run)],
          [[first, CARD_LOOK.text], [second, CARD_LOOK.text], [detail, themeVar("fgMuted")]],
          line,
        ));
      }
      lists.appendChild(grid);
      if (pairs.total > pairs.lines.length) lists.appendChild(moreLine(pairs.total - pairs.lines.length));
    }

    lists.style.display = lists.childElementCount ? "flex" : "none";
    lists.scrollTop = scrolled;
    node.setDirtyCanvas?.(true, false);
  };

  /**
   * Draw one report.
   *
   * @param {object} report - The report `createReportPanel` read.
   * @returns {void}
   */
  const update = (report) => {
    if (disposed || !report) return;
    // The run number and the report's content key the redraw.
    const signature = JSON.stringify([report.run, report.summary, report.tallies, report.facts, report.bodies]);
    if (drawnRun !== null && drawnRun === signature) return;
    drawnRun = signature;
    const scores = new Map((report.tallies ?? []).map((entry) => [entry.name, entry.value]));
    const axes = SCORES.filter(([key]) => scores.has(key)).map(([key, name]) => [key, name, Number(scores.get(key))]);
    drawRadar(axes.length >= 3 ? axes : SCORES.slice(0, EMPTY_AXES).map(([key, name]) => [key, name, null]));
    drawTable(report);
    drawChips(report);
    drawLists(report).catch((error) => console.error(`[${EXT_NAME}] Failed to draw the lists:`, error));
  };

  /**
   * Draw the empty frame a node shows before its first run.
   *
   * @returns {void}
   */
  const clear = () => {
    if (disposed) return;
    serial += 1;
    drawnRun = null;
    drawRadar(SCORES.slice(0, EMPTY_AXES).map(([key, name]) => [key, name, null]));
    drawTable(null);
    flagChips.textContent = "";
    phraseChips.textContent = "";
    flagBlock.style.display = "none";
    phraseBlock.style.display = "none";
    lists.textContent = "";
    lists.style.display = "none";
  };

  clear();

  return {
    element,
    update,
    clear,
    dispose() {
      disposed = true;
      serial += 1;
    },
  };
}

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_dataset_health_wrapped) return;
    proto.__was_dataset_health_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-dataset-health",
          height: PANEL_HEIGHT,
          minWidth: PANEL_WIDTH,
          facts: false,
          bodies: false,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the dataset's health:",
          sketch: () => createHealthSketch(this),
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the dataset's health:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the dataset's health:`, error);
      }
      return result;
    };
  },
});
