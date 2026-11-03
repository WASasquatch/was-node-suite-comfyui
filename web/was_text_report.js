/**
 * The report drawn on the text batch nodes.
 *
 * Every node listed below gets the pack's report panel: its summary, figures, facts and the
 * texts it published, such as what a replace changed or which captions were flagged.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";
import { LOOK } from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.TextReport";

const UI_WIDGET_NAME = "was_text_report_ui";
const UI_WIDGET_TYPE = "was_text_report";

// The widest fact name the reports write, which is what the fact column is opened at.
const LABEL_WIDTH = 96;

// Node id -> [panel height, the words drawn before the first run, the breakdown's bars].
// `bars.unit` follows each value; `bars.scale` is the value a full bar stands for, left out
// for the largest value in the breakdown.
const PANELS = {
  WASLoadTextBatch: [260, "run to read the folder", {}],
  WASTextReplaceWords: [360, "run to see the changes", { unit: "" }],
  WASCaptionStatistics: [440, "run to measure the captions", { unit: "%", scale: 100 }],
  WASTextListFilter: [260, "run to filter the list", {}],
  WASTextListSort: [240, "run to sort the list", {}],
  WASLoadDataset: [260, "run to load the dataset", {}],
  WASDatasetUpdate: [240, "run to update the dataset", {}],
  WASDatasetFilter: [300, "run to split the dataset", {}],
  WASDatasetMerge: [260, "run to join the datasets", {}],
};

/**
 * One sample row: its note, then its text with the marked span lit.
 *
 * @param {object} item - A row of the report, `{text, mark, note}`.
 * @returns {HTMLElement} The row.
 */
function sampleRow(item) {
  const row = document.createElement("div");
  row.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
  const text = String(item.text ?? "");
  row.title = item.note ? `${item.note}\n${text}` : text;
  if (item.note) {
    const note = document.createElement("span");
    note.style.cssText = `color:${LOOK.muted};margin-right:6px`;
    note.textContent = item.note;
    row.appendChild(note);
  }
  const mark = Array.isArray(item.mark) ? item.mark : null;
  if (!mark) {
    row.appendChild(document.createTextNode(text));
    return row;
  }
  const lit = document.createElement("span");
  lit.style.cssText = `color:${LOOK.text};background:color-mix(in srgb, ${LOOK.accent} 30%, transparent);`
    + "border-radius:2px";
  lit.textContent = text.slice(mark[0], mark[1]);
  row.append(document.createTextNode(text.slice(0, mark[0])), lit, document.createTextNode(text.slice(mark[1])));
  return row;
}

/**
 * A report's breakdown as one bar per row and its sample rows, for the report panel's sketch slot.
 *
 * @param {object} bars - `{unit, scale}` as `PANELS` holds them.
 * @returns {{element: HTMLElement, update: Function, clear: Function, dispose: Function}} The sketch.
 */
function createTallySketch(bars) {
  const unit = typeof bars.unit === "string" ? bars.unit : "";
  const scale = Number(bars.scale) > 0 ? Number(bars.scale) : 0;
  const element = document.createElement("div");
  element.style.cssText = "flex:0 0 auto;display:none;flex-direction:column;gap:6px;min-width:0";
  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;align-items:center;column-gap:8px;row-gap:3px;"
    + "grid-template-columns:fit-content(55%) minmax(36px,1fr) auto";
  const samples = document.createElement("div");
  samples.style.cssText = "display:flex;flex-direction:column;gap:2px;min-width:0";
  element.append(grid, samples);

  const clear = () => {
    grid.textContent = "";
    samples.textContent = "";
    element.style.display = "none";
  };

  return {
    element,
    update(report) {
      clear();
      const rows = (Array.isArray(report?.tallies) ? report.tallies : [])
        .filter((row) => typeof row?.name === "string" && Number.isFinite(Number(row?.value)));
      const items = (Array.isArray(report?.items) ? report.items : []).filter((item) => item?.text);
      if (!rows.length && !items.length) return;
      element.style.display = "flex";
      const most = scale || Math.max(1, ...rows.map((row) => Number(row.value)));
      for (const row of rows) {
        const value = Number(row.value);
        const name = document.createElement("span");
        name.style.cssText = `min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:${LOOK.text}`;
        name.textContent = row.name;
        name.title = row.name;
        const track = document.createElement("span");
        track.style.cssText = `display:block;height:6px;border-radius:3px;background:${LOOK.border};overflow:hidden`;
        const fill = document.createElement("span");
        fill.style.cssText = `display:block;height:100%;border-radius:3px;background:${LOOK.accent};`
          + `width:${Math.max(0, Math.min(100, (value / most) * 100)).toFixed(1)}%`;
        track.appendChild(fill);
        const figure = document.createElement("span");
        figure.style.cssText = `color:${LOOK.muted};text-align:right`;
        figure.textContent = `${Number.isInteger(value) ? value : value.toFixed(1)}${unit}`;
        grid.append(name, track, figure);
      }
      for (const item of items) samples.appendChild(sampleRow(item));
    },
    clear,
    dispose() {
      element.remove();
    },
  };
}

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const panel = PANELS[nodeData?.name];
    if (!panel) return;

    const proto = nodeType.prototype;
    if (proto.__was_text_report_wrapped) return;
    proto.__was_text_report_wrapped = true;
    const [height, emptyLabel, bars] = panel;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const report = createReportPanel(this, {
          className: "was-text-report",
          height,
          labelWidth: LABEL_WIDTH,
          emptyLabel,
          logName: EXT_NAME,
          failure: "Failed to read the text report:",
          sketch: () => createTallySketch(bars ?? {}),
        });
        appendInterfaceWidget(this, report, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        boundTextBoxes(this);

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            report.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the text report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the text report:`, error);
      }
      return result;
    };
  },
});
