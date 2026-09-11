/**
 * Which frames a sampler kept, drawn as a block per frame across the node.
 *
 * The blocks are worked out from the node's widgets, so changing a setting redraws the selection
 * without running anything.
 */

import { api } from "../../../scripts/api.js";
import { captureWheel, wheelPixels } from "./pointer.js";
import { PREVIEW_STATE, executionId } from "./preview.js";
import { createFigureTile, statusColour } from "./report_panel.js";
import { onRunEnded } from "./run_events.js";
import { RUN_LABELS, fetchRunResult } from "./run_result.js";
import { themeVar } from "./theme.js";

const LOG_NAME = "WASNodeSuite.FrameTimeline";

// Height in node units the panel opens at: the summary, the two figures, the fact rows and the
// strip, with nothing scrolling.
const PANEL_HEIGHT = 150;

// The narrowest the panel is worth drawing in, in node units. Below this the strip is too coarse
// to read a selection off and the fact rows start eliding. Without it the frontend refits the
// node to its sockets and leaves the panel standing outside it.
const PANEL_MIN_WIDTH = 260;

// Height of the block strip in CSS pixels.
const STRIP_HEIGHT = 26;

// The narrowest the fact-name column is drawn, in CSS pixels.
const LABEL_WIDTH = 52;

// Gap between a fact's name and its value, in CSS pixels.
const FACT_GAP = 10;

// The most blocks drawn. A longer clip buckets several frames into one block, which is what keeps
// a 4000 frame timeline from being 4000 elements wide.
const MAX_BLOCKS = 256;

/** Every strategy, in the order the schema lists them. */
export const STRATEGIES = ["uniform", "head", "center", "tail", "random", "every_nth"];

/** The strategies that answer consecutive frames. */
export const CONTIGUOUS = new Set(["head", "center", "tail"]);

const LCG_MULTIPLIER = 1664525;
const LCG_INCREMENT = 1013904223;

/**
 * Pick `count` distinct frames out of `total`, the same way python does.
 *
 * @param {number} total - Frames available.
 * @param {number} count - How many to pick.
 * @param {number} seed - Seed for the draw.
 * @returns {number[]} The chosen indices, ascending.
 */
export function scatter(total, count, seed) {
  // Reduced to 32 bits the same way python does, high half folded in so a seed that only differs
  // above bit 32 still draws differently.
  const whole = BigInt(seed);
  let state = Number((whole ^ (whole >> 32n)) & 0xffffffffn) >>> 0;
  const draw = () => {
    state = (Math.imul(LCG_MULTIPLIER, state) + LCG_INCREMENT) >>> 0;
    return state;
  };
  const pool = Array.from({ length: total }, (_, index) => index);
  for (let position = 0; position < count; position += 1) {
    const target = position + (draw() % (total - position));
    const held = pool[position];
    pool[position] = pool[target];
    pool[target] = held;
  }
  return pool.slice(0, count).sort((a, b) => a - b);
}

/**
 * Where a consecutive run starts and how many frames it holds.
 *
 * @param {number} total - Frames available.
 * @param {number} count - Frames wanted.
 * @param {string} strategy - One of `CONTIGUOUS`.
 * @returns {{start: number, taken: number}} The first index and how many follow it.
 */
export function frameSpan(total, count, strategy) {
  const taken = Math.max(1, Math.min(count, total));
  if (strategy === "head") return { start: 0, taken };
  if (strategy === "tail") return { start: total - taken, taken };
  return { start: Math.floor((total - taken) / 2), taken };
}

/**
 * Which positions of a pool a strategy keeps, ascending.
 *
 * @param {number} total - Positions available.
 * @param {number} count - The most to keep.
 * @param {string} strategy - One of `STRATEGIES`.
 * @param {number} seed - Seed, read only by `random`.
 * @returns {number[]} Positions, ascending, never more than `count` of them.
 */
function picked(total, count, strategy, seed) {
  const kept = Math.max(1, Math.min(count, total));
  if (CONTIGUOUS.has(strategy)) {
    const { start, taken } = frameSpan(total, kept, strategy);
    return Array.from({ length: taken }, (unused, index) => start + index);
  }
  if (strategy === "uniform") {
    if (kept === 1) return [Math.floor(total / 2)];
    // Rounded half away from zero, which is what python's round() does not do; every value here
    // is positive and non-half in practice.
    return Array.from({ length: kept }, (unused, index) => {
      const exact = (index * (total - 1)) / (kept - 1);
      const floor = Math.floor(exact);
      const rest = exact - floor;
      if (rest > 0.5) return floor + 1;
      if (rest < 0.5) return floor;
      // Python rounds a tie to the even neighbour.
      return floor % 2 === 0 ? floor : floor + 1;
    });
  }
  if (strategy === "random") return scatter(total, kept, seed);
  return Array.from({ length: kept }, (unused, index) => index);
}

/**
 * Which frames a strategy keeps, ascending.
 *
 * @param {number} total - Frames available.
 * @param {number} count - The most frames to keep.
 * @param {string} strategy - One of `STRATEGIES`.
 * @param {number} nth - Step between the frames a strategy may choose from, 1 for all.
 * @param {number} seed - Seed, read only by `random`.
 * @returns {number[]} Frame indices, ascending, never more than `count` of them.
 */
export function selection(total, count, strategy, nth, seed) {
  if (!(total > 0) || !STRATEGIES.includes(strategy)) return [];
  // The step thins the frames first and the strategy then chooses among what is left, so a
  // step means the same thing whichever strategy is reading it.
  const step = Math.max(1, nth);
  const pool = [];
  for (let index = 0; index < total; index += step) pool.push(index);
  return picked(pool.length, count, strategy, seed).map((position) => pool[position]);
}

/**
 * Build the panel a frame sampler draws its report and its timeline in.
 *
 * @param {object} node - The node the panel belongs to, for its widgets and its redraws.
 * @param {object} [options] - Overrides for a node whose widgets are not the sampler's four.
 * @param {function} [options.asked] - Given a widget reader, the values the pick depends on.
 * @param {function} [options.select] - Given the frame count and those values, the kept indices.
 * @returns {{element: HTMLElement, height: number, refresh: () => void, dispose: () => void}}
 *   The panel, for `appendInterfaceWidget`.
 */
export function createFrameTimelinePanel(node, options = {}) {
  const root = document.createElement("div");
  root.className = "was-frame-timeline";
  root.style.cssText = [
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    "display:flex",
    "flex-direction:column",
    "gap:6px",
    "overflow:hidden",
    "padding:8px 10px",
    "font:11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
    "line-height:1.4",
    `background:${themeVar("panelBg")}`,
    `color:${themeVar("fg")}`,
    `border:1px solid ${themeVar("border")}`,
    "border-radius:4px",
  ].join(";");

  const summary = document.createElement("div");
  summary.style.cssText = "font-weight:600;flex:0 0 auto";
  root.appendChild(summary);

  const counts = document.createElement("div");
  counts.style.cssText = `display:flex;gap:18px;flex:0 0 auto;color:${themeVar("fgMuted")}`;
  root.appendChild(counts);

  const strip = document.createElement("div");
  strip.style.cssText = [
    `height:${STRIP_HEIGHT}px`,
    "flex:0 0 auto",
    "display:flex",
    "gap:1px",
    "align-items:stretch",
    "border-radius:2px",
    "overflow:hidden",
  ].join(";");
  root.appendChild(strip);

  const scale = document.createElement("div");
  scale.style.cssText = "display:flex;justify-content:space-between;font-size:9px;flex:0 0 auto;"
    + `color:${themeVar("fgMuted")}`;
  root.appendChild(scale);

  const table = document.createElement("div");
  // Two grid columns shared by every row: the names as wide as the widest one, never wider than
  // half the panel, and the values taking what is left. The column re-measures when the node is
  // resized.
  table.style.cssText = "flex:1 1 auto;min-height:0;overflow:auto;display:grid;"
    + "grid-template-columns:fit-content(50%) minmax(0,1fr);align-items:baseline;"
    + `align-content:start;column-gap:${FACT_GAP}px;row-gap:2px`;
  root.appendChild(table);

  let disposed = false;
  // How long the clip was on the last run. The widgets say what to keep, but only a run says
  // what there was to keep it from, so the strip stays empty until one has happened.
  let total = 0;
  // What the widgets said when the report was drawn. Staleness is this against what they say
  // now, rather than a flag set when a change is noticed: the frontend's own seed control
  // assigns to `value` after every run without going through the widget's callback, so a flag
  // would stay clear while the strip quietly drew a different selection to the one reported.
  let shown = null;

  /**
   * One widget's value.
   *
   * @param {string} name - The widget's name.
   * @param {*} fallback - What to answer where the node carries no widget of that name.
   * @returns {*} The value.
   */
  const read = (name, fallback) => {
    const widget = (node.widgets ?? []).find((entry) => entry.name === name);
    return widget?.value ?? fallback;
  };

  // A node that picks frames some other way supplies both halves, so the strip lights what that
  // node will keep. Without them the sampler's four widgets are read and a node carrying none of
  // them would light a selection it never makes.
  const asked = options.asked
    ? () => options.asked(read)
    : () => ({
      count: Number(read("num_frames", 16)) || 1,
      strategy: String(read("strategy", "uniform")),
      nth: Number(read("nth", 1)) || 1,
      seed: Number(read("seed", 0)) || 0,
    });
  const pick = options.select
    ?? ((frames, values) =>
      selection(frames, values.count, values.strategy, values.nth, values.seed));

  /** Draw the block strip for what the widgets ask of the clip the last run saw. */
  const drawStrip = () => {
    strip.textContent = "";
    scale.textContent = "";
    if (!(total > 0)) {
      strip.style.background = themeVar("panelBg");
      return;
    }
    const now = asked();
    const stale = shown !== null && JSON.stringify(now) !== JSON.stringify(shown);
    const kept = new Set(pick(total, now));

    // One block per frame while they fit, otherwise a block per bucket of frames shaded by how
    // much of it was kept, so a long clip still shows where the selection falls.
    const blocks = Math.min(total, MAX_BLOCKS);
    const per = total / blocks;
    for (let index = 0; index < blocks; index += 1) {
      const from = Math.floor(index * per);
      const to = Math.max(from + 1, Math.floor((index + 1) * per));
      let hits = 0;
      for (let frame = from; frame < to; frame += 1) if (kept.has(frame)) hits += 1;
      const share = hits / (to - from);
      const block = document.createElement("div");
      block.style.cssText = [
        "flex:1 1 0",
        "min-width:0",
        `background:${themeVar(share > 0 ? "fg" : "border")}`,
        `opacity:${share > 0 ? (stale ? 0.45 : 0.35 + 0.65 * share) : stale ? 0.15 : 0.28}`,
        "border-radius:1px",
      ].join(";");
      block.title = to - from === 1 ? `frame ${from}` : `frames ${from}-${to - 1}`;
      strip.appendChild(block);
    }

    const first = document.createElement("span");
    first.textContent = "0";
    const middle = document.createElement("span");
    middle.textContent = stale ? "re-run to confirm" : `${kept.size} kept`;
    const last = document.createElement("span");
    last.textContent = String(total - 1);
    scale.append(first, middle, last);
  };

  /**
   * Draw one answer from the endpoint.
   *
   * @param {object} answer - The envelope `fetchRunResult` resolved to.
   * @returns {void}
   */
  const draw = (answer) => {
    if (disposed) return;
    const report = answer?.result;
    if (!report) {
      summary.textContent = answer?.label || RUN_LABELS[PREVIEW_STATE.WAITING] || "no run yet";
      summary.style.color = themeVar("fgMuted");
      counts.textContent = "";
      table.textContent = "";
      total = 0;
      shown = null;
      drawStrip();
      return;
    }
    summary.textContent = report.summary || "";
    summary.style.color = statusColour(report.status);

    counts.textContent = "";
    for (const { name, value } of report.counts ?? []) {
      if (name === "frames") total = Number(value) || 0;
      counts.appendChild(createFigureTile(name, value));
    }

    table.textContent = "";
    for (const { name, value } of report.facts ?? []) {
      const row = document.createElement("div");
      // The row's two spans sit in the table's own grid columns, so every name shares one
      // measured width and the values line up down the list.
      row.style.cssText = "display:contents";
      const key = document.createElement("span");
      key.style.cssText = `min-width:${LABEL_WIDTH}px;color:${themeVar("fgMuted")};`
        + "overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
      key.textContent = name;
      const val = document.createElement("span");
      val.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis;"
        + "white-space:nowrap";
      val.textContent = value;
      row.append(key, val);
      table.appendChild(row);
    }
    shown = asked();
    drawStrip();
    node.setDirtyCanvas?.(true, false);
  };

  let pending = false;
  let again = false;
  const refresh = async () => {
    if (disposed) return;
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
      console.error(`[${LOG_NAME}] Failed to read the sampler report:`, error);
    } finally {
      pending = false;
    }
  };

  /** Redraw the strip for a widget that just changed, without asking the server. */
  const preview = () => {
    if (disposed) return;
    // The statistics belong to the run that happened and the strip to what is asked for now,
    // so the strip is dimmed rather than the numbers being quietly rewritten to match it.
    drawStrip();
    node.setDirtyCanvas?.(true, false);
  };

  // Every widget here feeds the selection, and each is watched through its own `value` rather
  // than through its callback: the frontend's seed control writes the property directly, and a
  // callback-only watch would miss the one change the user did not make themselves.
  for (const widget of node.widgets ?? []) {
    let held = widget.value;
    try {
      Object.defineProperty(widget, "value", {
        configurable: true,
        enumerable: true,
        get: () => held,
        set: (next) => {
          const moved = next !== held;
          held = next;
          if (moved) preview();
        },
      });
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to watch ${widget.name}:`, error);
    }
  }

  const onExecuted = (event) => {
    const shown = event?.detail?.display_node;
    // The id the run publishes under, which for a node inside a subgraph is its whole path.
    if (shown !== undefined && String(shown) !== executionId(node)) return;
    refresh();
  };

  api.addEventListener("executed", onExecuted);
  const stopWatchingRuns = onRunEnded(() => refresh());

  // The rows are the only thing here that scrolls, and the panel takes every wheel gesture
  // over it, so the rows at either end leave the next tick doing nothing rather than zooming.
  const releaseWheel = captureWheel(root, (event) => {
    if (table.scrollHeight <= table.clientHeight || !table.contains(event.target)) return false;
    const before = table.scrollTop;
    table.scrollTop += wheelPixels(event, table).y;
    return table.scrollTop !== before;
  });

  refresh();

  return {
    element: root,
    height: PANEL_HEIGHT,
    // No ceiling worth naming: the node's own height is the bound, so dragging it taller gives
    // the rows the room rather than leaving a band of nothing below them.
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth: PANEL_MIN_WIDTH,
    refresh,
    dispose() {
      if (disposed) return;
      disposed = true;
      releaseWheel();
      api.removeEventListener?.("executed", onExecuted);
      if (typeof stopWatchingRuns === "function") stopWatchingRuns();
    },
  };
}
