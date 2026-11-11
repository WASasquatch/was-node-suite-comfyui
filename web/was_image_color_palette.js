/**
 * The palette Image Color Palette found, drawn as swatches on the node.
 *
 * The grid is painted from the `#rrggbb` codes the run report carries. A click copies the code
 * under the pointer.
 */

import { app } from "../../scripts/app.js";
import {
  STATUS, drawCell, outlineCell, parseColor, residualNote, tallyColours,
} from "./interface/colour_cell.js";
import { ICON, ICON_SIZE, drawFidelityGlyph } from "./interface/icons.js";
import { createReportPanel } from "./interface/report_panel.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onThemeChange, readTheme, themeVar } from "./interface/theme.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ColorPaletteUI";
const LOG_NAME = "WASNodeSuite.ColorPalette";
const SETTING_ID = "WAS.Analyze.ShowPalette";
const NODE_ID = "Image Color Palette";

const UI_WIDGET_NAME = "was_palette_readout_ui";
const UI_WIDGET_TYPE = "was_palette_readout";

// Height in node units the panel opens at: the summary, the tiles, two rows of swatches and the
// footer.
const PANEL_HEIGHT = 158;

// The side a swatch is drawn at when there is room, and the smallest it is allowed to become.
// Below the floor a 256 colour palette is a field of coloured pixels rather than swatches, so
// the grid is left taller than the panel and the panel scrolls it instead.
const TARGET_CELL = 44;
const MIN_CELL = 14;

// How long a copied code is named in the footer, in milliseconds.
const COPIED_MS = 1200;

// The name the node publishes its codes under.
const BODY_NAME = "palette";

// What the glyph claims over a grid of swatches, and what it claims over an empty one.
const PAINTED_FROM_CODES = Object.freeze({
  icon: ICON.EXACT,
  detail: "each swatch is the '#rrggbb' code the node returned, read as exact integers",
});
const NO_PALETTE = Object.freeze({
  icon: ICON.WARNING,
  detail: "no palette has arrived, so no swatch stands for a colour the node found",
});

/**
 * Whether the readout is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function enabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID);
    return typeof legacy === "boolean" ? legacy : true;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
    return true;
  }
}

/**
 * One named number of a report, or null when the report does not carry it.
 *
 * @param {object|null} report - A report from `run_result`.
 * @param {string} name - The count's name, as the node published it.
 * @returns {number|null} The value.
 */
function countOf(report, name) {
  const found = (report?.counts ?? []).find((entry) => entry.name === name);
  return found ? found.value : null;
}

/**
 * Where the swatches go, for the room there is and the number of them.
 *
 * @param {number} width - The room across, in element pixels.
 * @param {number} height - The room down, in element pixels.
 * @param {number} count - How many swatches there are.
 * @returns {{columns: number, rows: number, cell: number}} The layout. A count of zero answers
 *   no rows, so a caller never divides by it.
 */
function gridFor(width, height, count) {
  const total = Math.max(0, Math.trunc(count));
  if (!(width > 0) || total === 0) return { columns: 1, rows: 0, cell: MIN_CELL };
  const room = Math.max(0, height);
  let last = null;
  for (let cell = Math.max(MIN_CELL, TARGET_CELL); cell >= MIN_CELL; cell -= 1) {
    const across = Math.max(1, Math.min(total, Math.floor(width / cell)));
    const rows = Math.ceil(total / across);
    // Spread over the rows that many needs rather than filling each one to the width, so 16
    // swatches over two rows are 8 and 8 instead of 12 and 4.
    last = { columns: Math.ceil(total / rows), rows, cell };
    if (rows * cell <= room) return last;
  }
  return last;
}

/**
 * Build the swatch grid a report panel draws between its counts and its rows.
 *
 * @param {object} node - The node the grid belongs to, for its redraws.
 * @returns {{element: HTMLElement, update: (report: object) => void, clear: () => void,
 *   dispose: () => void}} The band, in the shape `createReportPanel` takes as its sketch.
 */
export function createSwatchGrid(node) {
  const element = document.createElement("div");
  element.style.cssText = "flex:1 1 auto;min-height:0;display:flex;flex-direction:column;gap:4px";

  const scroller = document.createElement("div");
  scroller.style.cssText = "flex:1 1 auto;min-height:0;overflow:auto";
  element.appendChild(scroller);

  const grid = document.createElement("canvas");
  grid.style.cssText = "display:block;width:100%";
  scroller.appendChild(grid);

  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;align-items:center;gap:6px;flex:0 0 auto;font-size:9px";
  element.appendChild(footer);

  const glyph = document.createElement("canvas");
  glyph.style.cssText = `width:${ICON_SIZE}px;height:${ICON_SIZE}px;flex:0 0 auto`;
  const note = document.createElement("span");
  note.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + `white-space:nowrap;color:${themeVar("fgMuted")}`;
  footer.append(glyph, note);

  let disposed = false;
  // What the last report carried: the codes, their parses and where each cell was drawn, so a
  // pointer is answered without parsing the body again.
  let codes = [];
  let parsed = [];
  let cells = [];
  let footerText = "";
  let copiedHandle = 0;

  /** Draw the fidelity glyph, which is exact: a swatch is the code itself. */
  const drawGlyph = () => {
    drawFidelityGlyph(glyph, codes.length ? PAINTED_FROM_CODES : NO_PALETTE);
  };

  /** Draw the swatches at whatever room the panel currently has. */
  const drawGrid = () => {
    if (disposed) return;
    const theme = readTheme();
    const width = scroller.clientWidth;
    if (!(width > 0)) return;
    const layout = gridFor(width, scroller.clientHeight, parsed.length);
    const height = Math.max(1, layout.rows * layout.cell);
    // Written in element pixels first, so the scroller learns how tall the grid is before the
    // backing store is sized against the ratio that height is measured at.
    grid.style.height = `${height}px`;

    const ratio = surfaceRatio(grid);
    const w = Math.max(1, Math.round(width * ratio));
    const h = Math.max(1, Math.round(height * ratio));
    if (grid.width !== w || grid.height !== h) { grid.width = w; grid.height = h; }
    const ctx = grid.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.scale(ratio, ratio);

    cells = [];
    const markFont = `${Math.max(7, Math.round(layout.cell / 3))}px sans-serif`;
    for (let index = 0; index < parsed.length; index += 1) {
      const rect = {
        x: (index % layout.columns) * layout.cell,
        y: Math.floor(index / layout.columns) * layout.cell,
        width: layout.cell,
        height: layout.cell,
      };
      cells.push(rect);
      drawCell(ctx, ratio, rect, parsed[index], { markColour: theme.warning, markFont });
      outlineCell(ctx, rect, theme.border);
    }
    node.setDirtyCanvas?.(true, false);
  };

  /**
   * Read one report and draw the palette it carries.
   *
   * @param {object|null} report - The report `createReportPanel` is drawing.
   * @returns {void}
   */
  const update = (report) => {
    if (disposed) return;
    const carried = (report?.bodies ?? []).find((entry) => entry.name === BODY_NAME);
    codes = (carried?.text ?? "").split("\n").map((line) => line.trim()).filter(Boolean);
    parsed = codes.map((code) => parseColor(code));

    // The node states how many colours it found separately from the codes it could carry, so a
    // palette larger than one report holds says so rather than looking like a smaller palette.
    const stated = countOf(report, "colours") ?? codes.length;
    footerText = [
      codes.length < stated ? `${codes.length} of ${stated} shown` : `${codes.length} colours`,
      residualNote(tallyColours(parsed), "code", "codes"),
    ].filter(Boolean).join(", ");
    note.textContent = footerText;
    drawGrid();
    drawGlyph();
  };

  /**
   * Which swatch is under a pointer event.
   *
   * @param {PointerEvent|MouseEvent} event - The event on the grid.
   * @returns {number} The index, or -1 for a point on no cell.
   */
  const cellAt = (event) => {
    const box = grid.getBoundingClientRect();
    if (!(box.width > 0) || !(box.height > 0)) return -1;
    // The client rectangle carries the graph's zoom and the cells were laid out in element
    // pixels, so the point is divided back before it is compared with them.
    const x = (event.clientX - box.x) * (grid.clientWidth / box.width);
    const y = (event.clientY - box.y) * (grid.clientHeight / box.height);
    return cells.findIndex(
      (rect) => x >= rect.x && x < rect.x + rect.width && y >= rect.y && y < rect.y + rect.height,
    );
  };

  const onMove = (event) => {
    const index = cellAt(event);
    if (index < 0) { grid.title = ""; return; }
    grid.title = parsed[index]?.status === STATUS.COLOUR
      ? codes[index]
      : `${codes[index]} is not a colour this panel can read`;
  };
  const onLeave = () => { grid.title = ""; };
  const onClick = (event) => {
    const index = cellAt(event);
    if (index < 0) return;
    const code = codes[index];
    try {
      navigator.clipboard?.writeText?.(code);
      note.textContent = `copied ${code}`;
      window.clearTimeout(copiedHandle);
      copiedHandle = window.setTimeout(() => {
        if (!disposed) note.textContent = footerText;
      }, COPIED_MS);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to copy ${code}:`, error);
    }
  };
  const onWheel = (event) => {
    // A grid taller than its room scrolls itself, so the gesture is kept here and the panel
    // around it never sees it.
    if (scroller.scrollHeight > scroller.clientHeight) event.stopPropagation();
  };

  grid.addEventListener("pointermove", onMove);
  grid.addEventListener("pointerleave", onLeave);
  grid.addEventListener("click", onClick);
  scroller.addEventListener("wheel", onWheel, { passive: true });

  const observer = typeof ResizeObserver === "function"
    ? new ResizeObserver(() => drawGrid())
    : null;
  observer?.observe(scroller);
  const stopWatching = watchSurfaceRatio(grid, () => { drawGrid(); drawGlyph(); });
  // The grid and the glyph are canvases, which take literal colours, so a palette change draws
  // them again.
  const stopTheme = onThemeChange(() => { drawGrid(); drawGlyph(); });

  drawGlyph();

  return {
    element,
    update,
    clear() {
      codes = [];
      parsed = [];
      cells = [];
      footerText = "";
      note.textContent = "";
      drawGrid();
      drawGlyph();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      window.clearTimeout(copiedHandle);
      grid.removeEventListener("pointermove", onMove);
      grid.removeEventListener("pointerleave", onLeave);
      grid.removeEventListener("click", onClick);
      scroller.removeEventListener("wheel", onWheel);
      observer?.disconnect();
      stopWatching();
      stopTheme();
    },
  };
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Analyze", "Show the palette swatches"],
      name: "Draw the palette swatches on Image Color Palette",
      tooltip:
        "Draw the colours the node found as a grid of swatches on the node, naming the hex code "
        + "of the one under the pointer and copying it on a click. The chart output is unchanged "
        + "either way. This applies to nodes added after the setting changes, so a reload shows "
        + "it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_palette_readout_wrapped) return;
    proto.__was_palette_readout_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const node = this;
        const panel = createReportPanel(node, {
          className: "was-palette-readout",
          // The one fact the node publishes says how much of the palette the report could
          // carry, which the footer under the swatches says in the swatches' own terms.
          facts: false,
          // The grid is drawn from the same codes the body carries, and drawing both leaves
          // the swatches half the panel.
          bodies: false,
          height: PANEL_HEIGHT,
          emptyLabel: "No palette yet",
          sketch: () => createSwatchGrid(node),
          logName: LOG_NAME,
          failure: "Failed to read the palette report:",
        });
        appendInterfaceWidget(node, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the palette readout:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the palette readout:`, error);
      }
      return result;
    };
  },
});
