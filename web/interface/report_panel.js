/**
 * The report a node published, drawn on the node it belongs to.
 *
 * `createReportPanel` draws one `run_result` envelope: a summary line, a row of counts and the
 * fact rows. Counts and facts arrive as arrays of `{name, value}`.
 */

import { api } from "../../../scripts/api.js";
import { createListing } from "./listing.js";
import { captureWheel, wheelPixels } from "./pointer.js";
import { PREVIEW_STATE, executionId } from "./preview.js";
import { onNodeFinished, onRunEnded } from "./run_events.js";
import { RUN_LABELS, fetchRunResult, fetchRunResultPage } from "./run_result.js";
import { themeVar } from "./theme.js";

const LOG_NAME = "WASNodeSuite.ReportPanel";

// The figure and its label, in CSS pixels, for every tile in the pack.
const FIGURE_SIZE = 12;
const LABEL_SIZE = 8;

// Height in node units a panel opens at when the caller names none.
const DEFAULT_HEIGHT = 116;

// Gap between the count tiles, in CSS pixels.
const TILE_GAP = 18;

// Overflow values that make a region the one a wheel scrolls.
const SCROLLS = ["auto", "scroll", "overlay"];

// Height of a frame tab strip, and of a tab inside it, in CSS pixels.
const TAB_STRIP_HEIGHT = 22;
const TAB_HEIGHT = 20;

/**
 * The two shapes a report panel is drawn in.
 *
 * `column` scrolls the fact rows inside the panel; `flow` scrolls the whole panel.
 */
const LAYOUTS = {
  column: { padding: "8px 10px", lineHeight: 1.4, labelWidth: 52, factGap: 10, blockGap: 6 },
  flow: { padding: "6px 8px", lineHeight: 1.45, labelWidth: 62, factGap: 8, blockGap: 4 },
};

/**
 * One figure over its name, drawn large enough to read at a glance.
 *
 * @param {string} name - What the figure is called, drawn under it in uppercase.
 * @param {string|number} value - The figure itself, already written the way it is to be read.
 * @param {string} [hint] - A sentence shown on hover, for a figure that needs one.
 * @returns {HTMLElement} The tile, for the caller to append.
 */
export function createFigureTile(name, value, hint = "") {
  const tile = document.createElement("div");
  tile.style.cssText = "display:flex;flex-direction:column;line-height:1.15";
  if (hint) tile.title = hint;
  const figure = document.createElement("span");
  figure.style.cssText = `font-size:${FIGURE_SIZE}px;font-weight:600;color:${themeVar("fg")}`;
  figure.textContent = String(value);
  const label = document.createElement("span");
  label.style.cssText = `font-size:${LABEL_SIZE}px;letter-spacing:0.08em;text-transform:uppercase;`
    + `color:${themeVar("fgMuted")}`;
  label.textContent = name;
  tile.append(figure, label);
  return tile;
}

/**
 * A byte count in the largest unit that keeps it readable.
 *
 * Whole bytes below a kilobyte and one decimal above it.
 *
 * @param {number} bytes - The count.
 * @returns {string} The count with its unit, such as `16.0 MB`.
 */
export function readableBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) return "";
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit === 0 ? Math.trunc(value) : value.toFixed(1)} ${units[unit]}`;
}

/**
 * A row of numbered tabs, one per frame of a batch.
 *
 * Hidden below two frames.
 *
 * @param {(index: number) => void} onPick - Called with the frame a tab was clicked for.
 * @param {object} [options] - How the strip is drawn.
 * @param {boolean} [options.overlay] - Lay the strip over what is below it rather than in a row
 *   of its own, for a panel whose whole height is a picture.
 * @returns {{element: HTMLElement, draw: (frames: number, current: number, measured?: number)
 *   => void}} The strip, and the redraw its owner calls on every repaint.
 */
export function createFrameTabs(onPick, options = {}) {
  const overlay = options.overlay === true;
  const strip = document.createElement("div");
  strip.style.cssText = [
    ...(overlay
      ? ["position:absolute", "left:0", "right:0", "top:0", "z-index:2", "padding:2px 3px"]
      : ["flex:0 0 auto"]),
    `height:${TAB_STRIP_HEIGHT}px`,
    "min-height:0",
    "box-sizing:border-box",
    "display:none",
    "gap:4px",
    "align-items:center",
    "overflow-x:auto",
    "overflow-y:hidden",
    "scrollbar-width:thin",
  ].join(";");

  const draw = (frames, current, measured = frames) => {
    const count = Math.max(0, Math.trunc(frames));
    strip.style.display = count > 1 ? "flex" : "none";
    // Drawn over the picture, so the tabs need something behind them to stay legible against
    // whatever the picture happens to be.
    if (overlay) {
      strip.style.background = "linear-gradient(to bottom, rgba(0,0,0,0.62), rgba(0,0,0,0))";
    }
    if (count <= 1) {
      strip.textContent = "";
      return;
    }
    if (strip.childElementCount !== count) {
      strip.textContent = "";
      for (let index = 0; index < count; index += 1) {
        const tab = document.createElement("button");
        tab.type = "button";
        tab.textContent = String(index + 1);
        tab.style.cssText = [
          "flex:0 0 auto",
          "min-width:22px",
          `height:${TAB_HEIGHT}px`,
          "padding:0 6px",
          "font:10px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
          "border-radius:3px",
          "cursor:pointer",
        ].join(";");
        tab.addEventListener("pointerdown", (event) => event.stopPropagation());
        tab.addEventListener("click", (event) => {
          event.stopPropagation();
          onPick(index);
        });
        strip.appendChild(tab);
      }
    }
    Array.from(strip.children).forEach((tab, index) => {
      const on = index === current;
      tab.style.background = themeVar(on ? "fg" : "panelBg");
      tab.style.color = themeVar(on ? "panelBg" : (index < measured ? "fgMuted" : "border"));
      tab.style.border = `1px solid ${themeVar(on ? "fg" : "border")}`;
    });
  };

  return { element: strip, draw };
}

/**
 * The colour a report's status is drawn in.
 *
 * @param {string} status - `run_result`'s status for the report.
 * @returns {string} A `var()` reference to a palette property, the ordinary foreground for
 *   anything but a warning or an error.
 */
export function statusColour(status) {
  if (status === "error") return themeVar("error");
  if (status === "warning") return themeVar("warning");
  return themeVar("fg");
}

/**
 * Whether one region of a panel holds more than it has room for.
 *
 * @param {Element} region - The element under the pointer, or one above it.
 * @returns {boolean} True when a wheel over it scrolls it.
 */
function scrollable(region) {
  if (!(region?.scrollHeight > region.clientHeight)) return false;
  return SCROLLS.includes(getComputedStyle(region).overflowY);
}

/**
 * The report inside whatever the caller handed over.
 *
 * @param {object|null} answer - A `fetchRunResult` envelope, or a bare report.
 * @returns {object|null} The report, or null when there is none to draw.
 */
function reportOf(answer) {
  if (!answer || typeof answer !== "object") return null;
  if ("result" in answer || "state" in answer) return answer.result || null;
  return answer.summary || answer.counts || answer.facts || answer.bodies || answer.status
    ? answer : null;
}

/**
 * Build the panel a node draws its run report in.
 *
 * @param {object} node - The node the panel belongs to, for its id and its redraws.
 * @param {object} [options] - How the report is drawn.
 * @param {Function} [options.onAnswer] - Called with each report before it is drawn,
 *   and with null where the node has not run, for a node reading its own figures back.
 * @param {boolean} [options.summary] - Draw the summary line. On by default, and it is where
 *   the words for a node that has not reported are drawn.
 * @param {boolean} [options.tiles] - Draw the counts as figure tiles. Off draws them as one
 *   line of `name value` pairs.
 * @param {boolean} [options.facts] - Draw the fact rows. On by default.
 * @param {boolean} [options.bodies] - Draw the texts the report carries. On by default. Off
 *   for a panel whose sketch draws the same body itself.
 * @param {boolean} [options.footer] - Draw the summary, the counts and the facts under the
 *   bodies rather than above them, so the value is what the panel opens on.
 * @param {Function} [options.sketch] - A factory called once, answering
 *   `{element, update, clear, dispose}` in the same shape this panel answers. Its element is
 *   drawn between the counts and the facts, its `update` takes the report on every draw, and
 *   its `clear` and `dispose` are called with the panel's own.
 * @param {number} [options.height] - Height in node units the panel opens at.
 * @param {number} [options.maxHeight] - The tallest it may be dragged to. Unbounded by default,
 *   so the node's spare room reaches the panel.
 * @param {number} [options.minWidth] - The narrowest the panel is worth drawing in, in node
 *   units. Left out, the node may be collapsed under it.
 * @param {string} [options.emptyLabel] - Words drawn when the node has published no report.
 * @param {string} [options.layout] - `column` or `flow`, the two shapes above.
 * @param {number} [options.labelWidth] - Narrowest the fact-name column is drawn, in CSS
 *   pixels. The column widens to the longest name, up to half the panel.
 * @param {number} [options.factGap] - Gap between a fact's name and its value, in CSS pixels.
 * @param {string} [options.padding] - The panel's own padding.
 * @param {number} [options.lineHeight] - Line height for the whole panel.
 * @param {boolean} [options.live] - Listen for the node's own `executed` event, which is what
 *   makes a node reporting once per iteration read live. On by default.
 * @param {string} [options.className] - The class put on the panel, which no stylesheet reads
 *   and which is there to be found in the document while checking one.
 * @param {string} [options.logName] - The name a failure is logged under.
 * @param {string} [options.failure] - The sentence a failure is logged with.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, minWidth: number,
 *   update: Function, clear: Function, refresh: Function, dispose: Function}} The panel, for
 *   `appendInterfaceWidget`.
 */
export function createReportPanel(node, options = {}) {
  const flow = options.layout === "flow";
  const shape = flow ? LAYOUTS.flow : LAYOUTS.column;
  const withSummary = options.summary !== false;
  const withTiles = options.tiles !== false;
  const withFacts = options.facts !== false;
  const withBodies = options.bodies !== false;
  const asFooter = options.footer === true;
  const live = options.live !== false;
  const labelWidth = Number(options.labelWidth) > 0 ? Number(options.labelWidth) : shape.labelWidth;
  const factGap = Number.isFinite(options.factGap) ? options.factGap : shape.factGap;
  const blockGap = shape.blockGap;
  const padding = options.padding || shape.padding;
  const lineHeight = options.lineHeight || shape.lineHeight;
  const height = Number(options.height) > 0 ? Number(options.height) : DEFAULT_HEIGHT;
  const maxHeight = Number(options.maxHeight) > 0 ? Number(options.maxHeight)
    : Number.MAX_SAFE_INTEGER;
  const minWidth = Number(options.minWidth) > 0 ? Number(options.minWidth) : 0;
  const emptyLabel = options.emptyLabel ?? RUN_LABELS[PREVIEW_STATE.WAITING];
  const logName = options.logName || LOG_NAME;
  const failure = options.failure || "Failed to read the run report:";
  const onAnswer = typeof options.onAnswer === "function" ? options.onAnswer : null;

  const root = document.createElement("div");
  root.className = options.className || "was-report-panel";
  root.style.cssText = [
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    ...(flow
      ? ["overflow:auto"]
      : ["display:flex", "flex-direction:column", `gap:${blockGap}px`, "overflow:hidden"]),
    `padding:${padding}`,
    "font:11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
    `line-height:${lineHeight}`,
    "border-radius:4px",
    `background:${themeVar("panelBg")}`,
    `color:${themeVar("fg")}`,
    `border:1px solid ${themeVar("border")}`,
  ].join(";");

  // A footer draws the same three blocks under the bodies, so they are appended to their
  // own element rather than straight to the panel.
  const head = asFooter ? document.createElement("div") : root;
  if (asFooter) {
    head.style.cssText = "flex:0 0 auto;display:flex;flex-direction:column;"
      + `gap:${blockGap}px;margin-top:${blockGap}px`;
  }

  const summary = document.createElement("div");
  summary.style.cssText = "font-weight:600"
    + (flow || asFooter ? `;margin-bottom:${asFooter ? 0 : blockGap}px` : ";flex:0 0 auto");
  if (withSummary) head.appendChild(summary);

  // The numbers that say what happened, drawn apart from the rows so the panel is scanned
  // rather than read.
  const counts = document.createElement("div");
  counts.style.cssText = (withTiles
    ? `display:flex;flex-wrap:wrap;gap:${TILE_GAP}px`
      + (flow ? `;margin-bottom:${blockGap}px` : ";flex:0 0 auto")
    : (flow ? `margin-bottom:${blockGap}px` : "flex:0 0 auto"))
    + `;color:${themeVar("fgMuted")}`;
  head.appendChild(counts);

  const band = typeof options.sketch === "function" ? options.sketch() : null;
  if (band?.element) head.appendChild(band.element);

  const table = document.createElement("div");
  // Two grid columns shared by every row: the names as wide as the widest one, never wider
  // than half the panel, and the values taking what is left. The column re-measures when the
  // node is resized.
  const factColumns = "grid-template-columns:fit-content(50%) minmax(0,1fr);align-items:baseline";
  // Tall enough for its own rows and no taller, so the listing under it keeps the rest of the
  // panel rather than the two sharing the height evenly. A row is its own content and is never
  // squeezed, so a fact is never cut across the middle.
  table.style.cssText = `display:grid;${factColumns};column-gap:${factGap}px;row-gap:2px`
    + (flow ? "" : ";flex:0 0 auto;grid-auto-rows:min-content");
  if (withFacts) head.appendChild(table);

  const bodies = document.createElement("div");
  // Takes the panel's spare height only while it holds a text, so a sketch band above it keeps
  // that height where the report carries none. A footer keeps the height either way, which is
  // what holds it against the bottom edge.
  const idleFlex = asFooter ? "1 1 auto" : "0 0 auto";
  bodies.style.cssText = `flex:${idleFlex};min-height:0;overflow:auto;display:flex;`
    + "flex-direction:column;gap:6px";
  if (withBodies) root.appendChild(bodies);
  if (asFooter) root.appendChild(head);

  let disposed = false;

  // The listings the last report was drawn with, each holding a page reader of its own.
  const drawn = [];

  /**
   * Drop the listings the last report was drawn with.
   *
   * @returns {void}
   */
  const release = () => {
    for (const listing of drawn.splice(0)) listing.dispose?.();
  };

  /**
   * Read a range of lines from one body of the report this node published.
   *
   * @param {number} index - Which body, counting from zero in the order the report carries them.
   * @param {number} start - The first line wanted, counting from zero.
   * @param {number} wanted - How many lines to ask for.
   * @returns {Promise<object|null>} The page, or null where it could not be read.
   */
  const readPage = async (index, start, wanted) => {
    if (disposed) return null;
    const answer = await fetchRunResultPage(node, index, start, wanted);
    return answer?.page ?? null;
  };

  /**
   * Draw one answer from the endpoint.
   *
   * `fetchRunResult` answers an envelope, `{state, label, result}`, rather than the report.
   *
   * @param {object|null} answer - The envelope, or a bare report.
   * @returns {void}
   */
  const draw = (answer) => {
    if (disposed) return;
    const report = reportOf(answer);
    if (onAnswer) {
      try {
        onAnswer(report);
      } catch (error) {
        console.error(`[${logName}] Failed to read the report back onto the node:`, error);
      }
    }
    if (!report) {
      // The caller's words stand in only for a node that has published nothing; a
      // disconnected or failed fetch keeps the envelope's own label.
      const waiting = !answer || answer.state === PREVIEW_STATE.WAITING;
      summary.textContent = (waiting ? emptyLabel : answer?.label)
        || answer?.label || emptyLabel || "no run yet";
      summary.style.color = themeVar("fgMuted");
      counts.textContent = "";
      table.textContent = "";
      release();
      bodies.textContent = "";
      bodies.style.flex = idleFlex;
      band?.clear?.();
      node.setDirtyCanvas?.(true, false);
      return;
    }
    summary.textContent = report.summary || "";
    summary.style.color = statusColour(report.status);

    counts.textContent = "";
    const named = report.counts ?? [];
    if (withTiles) {
      for (const { name, value } of named) counts.appendChild(createFigureTile(name, value));
    } else {
      counts.textContent = named.map((entry) => `${entry.name} ${entry.value}`).join("   ");
    }

    band?.update?.(report);

    table.textContent = "";
    for (const { name, value } of withFacts ? report.facts ?? [] : []) {
      const row = document.createElement("div");
      // The row's two spans sit in the table's own grid columns, so every name shares one
      // measured width and the values line up down the list.
      row.style.cssText = "display:contents";
      const key = document.createElement("span");
      key.style.cssText = `min-width:${labelWidth}px;color:${themeVar("fgMuted")};overflow:hidden;`
        + "text-overflow:ellipsis;white-space:nowrap";
      key.textContent = name;
      const val = document.createElement("span");
      val.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis"
        + (flow ? ";white-space:pre-wrap" : ";white-space:nowrap");
      val.textContent = value;
      row.append(key, val);
      table.appendChild(row);
    }

    release();
    bodies.textContent = "";
    (withBodies ? report.bodies ?? [] : []).forEach((carried, index) => {
      const { name, text, whole, lines, offset } = carried;
      if (!text) return;
      const listing = createListing(text, {
        lines,
        whole,
        offset,
        name,
        run: report.run,
        page: (start, wanted) => readPage(index, start, wanted),
      });
      drawn.push(listing);
      const block = document.createElement("div");
      block.style.cssText = "display:flex;flex-direction:column;gap:2px;min-height:0";
      if (name) {
        const label = document.createElement("div");
        label.style.cssText = `color:${themeVar("fgMuted")};font-size:9px`;
        // A body longer than the report carries arrives as a piece of itself, and where the
        // rest of it cannot be read a page at a time the heading counts what is drawn rather
        // than letting the total stand for it.
        label.textContent = whole === false && listing.dataset.wasPaged !== "1"
          ? `${name}, ${String(text).split("\n").length} drawn`
          : name;
        block.appendChild(label);
      }
      block.appendChild(listing);
      bodies.appendChild(block);
    });
    bodies.style.flex = bodies.childElementCount > 0 ? "1 1 auto" : idleFlex;
    node.setDirtyCanvas?.(true, false);
  };

  let pending = false;
  let again = false;
  const refresh = async () => {
    if (disposed) return;
    // Asked again while a read is in flight, the answer already on its way describes an older
    // run, so the ask is remembered and served after it rather than dropped. The last thing a
    // node reports and the end of the run land within a few milliseconds of each other, and
    // dropping the second leaves the panel showing the state before the run finished.
    if (pending) {
      again = true;
      return;
    }
    pending = true;
    try {
      do {
        again = false;
        draw(await fetchRunResult(node));
      } while (again && !disposed);
    } catch (error) {
      console.error(`[${logName}] ${failure}`, error);
    } finally {
      pending = false;
    }
  };

  // A node that reports more than once in a run sends an `executed` event with each report, and
  // `display_node` is the node on the canvas rather than the clone that ran, which is what lets
  // a panel recognise an iteration of its own loop.
  const onExecuted = (event) => {
    const shown = event?.detail?.display_node;
    if (shown !== undefined && String(shown) !== executionId(node)) return;
    refresh();
  };
  if (live) api.addEventListener("executed", onExecuted);

  const stopFinished = onNodeFinished(node, () => refresh());
  const stopEnded = onRunEnded(() => refresh());

  // A region under the pointer scrolls while it has somewhere left to go. Once it has not, the
  // gesture is the graph's and zooms the node under the pointer.
  const releaseWheel = captureWheel(root, (event) => {
    const from = event.target instanceof Element ? event.target : root;
    for (let region = from; region; region = region.parentElement) {
      if (scrollable(region)) {
        const before = region.scrollTop;
        region.scrollTop += wheelPixels(event, region).y;
        return region.scrollTop !== before;
      }
      if (region === root) break;
    }
    return false;
  });

  refresh();

  return {
    element: root,
    height,
    maxHeight,
    minWidth,
    update: draw,
    clear: () => draw(null),
    refresh,
    dispose() {
      if (disposed) return;
      disposed = true;
      if (live) api.removeEventListener?.("executed", onExecuted);
      if (typeof stopFinished === "function") stopFinished();
      if (typeof stopEnded === "function") stopEnded();
      releaseWheel();
      release();
      band?.dispose?.();
    },
  };
}
