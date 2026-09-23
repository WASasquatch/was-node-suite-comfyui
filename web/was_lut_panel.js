/**
 * What a LUT node built, drawn on the node.
 *
 * The transfer curves come from the neutral band of the graded reference strip, so the plot
 * is the table's own response rather than a second calculation of it.
 */

import { app } from "../../scripts/app.js";
import { readPixels } from "./interface/image_metrics.js";
import { PREVIEW_STATE, fetchOutputPreview } from "./interface/preview.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onRunEnded } from "./interface/run_events.js";
import { fetchRunResult } from "./interface/run_result.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.LUTPanel";
const SETTING_ID = "WAS.LUT.ShowInterface";

const UI_WIDGET_NAME = "was_lut_ui";
const UI_WIDGET_TYPE = "was_lut_panel";

// Slot the graded chart is published under. `modules/interface/lut_report.py` names the same.
const STRIP_SLOT = "lut_strip";

// Every node this draws on. Apply LUT is deliberately absent: it already carries the before
// and after band, and two panels on one node leave neither of them room.
const NODES = ["WASNSLoadLUT", "WASNSCombineLUT", "WASNSSaveLUT", "WASLUTFromReference"];

// Of those, the two that build a table with no picture anywhere near them, which get the
// curves and the chart. The rest state what the table is and leave looking to a preview node.
const NODES_WITH_CHART = ["WASNSLoadLUT", "WASNSCombineLUT"];

// Rows of the published strip, matching `modules/image/lut_preview.py`. The ramp is read for
// the curves and the two bands below it are drawn as the chart.
const RAMP_ROWS = 16;
const HUE_ROWS = 16;
const PATCH_ROWS = 16;
const STRIP_ROWS = RAMP_ROWS + HUE_ROWS + PATCH_ROWS;

const UI_HEIGHT = 190;
const FACTS_HEIGHT = 44;
const MAX_UI_HEIGHT = 320;
const MIN_UI_WIDTH = 200;
const PAD = 10;
const FACT_LINE = 13;
const CHART_HEIGHT = 26;

/**
 * Read whether the panel is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function panelEnabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID, true);
    if (typeof legacy === "boolean") return legacy;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
  }
  return true;
}

/**
 * Build the panel for one LUT node.
 *
 * @param {object} node - The node the panel belongs to.
 * @param {boolean} withChart - Draw the curves and the chart, not just the facts.
 * @returns {object} A panel for `appendInterfaceWidget`, with the hooks the node chains onto.
 */
function createLutPanel(node, withChart) {
  // What each channel's transfer curve is drawn in, as pairs so no table of bare channel
  // letters sits at module level where a reader looks for node ids.
  const strokes = [["r", "#ff6b6b"], ["g", "#6bd66b"], ["b", "#6b9dff"]];

  const element = document.createElement("div");
  element.style.cssText = "width:100%;height:100%;position:relative;overflow:hidden;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;height:100%;display:block;";
  element.appendChild(canvas);

  const state = { facts: [], summary: "", curves: null, chart: null, disposed: false, painting: false };
  let releaseRatio = null;
  let releaseRun = null;

  /**
   * Repaint on the next frame, once however many changes arrive this one.
   *
   * @returns {void}
   */
  function schedulePaint() {
    if (state.painting || state.disposed) return;
    state.painting = true;
    requestAnimationFrame(() => {
      state.painting = false;
      if (!state.disposed) paint();
    });
  }

  /**
   * Read the facts, and the strip the curves are taken off.
   *
   * @returns {Promise<void>} Resolves once the panel has what it can draw.
   */
  async function load() {
    try {
      const answer = await fetchRunResult(node);
      const report = answer?.result ?? null;
      // The route answers facts as `{name, value}` rows, in the order they were published.
      state.facts = (report?.facts ?? []).map(({ name, value }) => [String(name), String(value)]);
      state.summary = report?.summary ?? "";
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read what the table is:`, error);
    }
    if (withChart) {
      try {
        const answer = await fetchOutputPreview(node, STRIP_SLOT);
        if (answer?.state === PREVIEW_STATE.READY && answer.image) {
          readStrip(answer.image, answer.sourceWidth, answer.sourceHeight);
        } else {
          state.curves = null;
          state.chart = null;
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to read the graded strip:`, error);
      }
    }
    if (!state.disposed) schedulePaint();
  }

  /**
   * Take the transfer curves and the chart out of one graded strip.
   *
   * @param {CanvasImageSource} image - The published strip.
   * @param {number} width - Its width in pixels.
   * @param {number} height - Its height in pixels.
   * @returns {void}
   */
  function readStrip(image, width, height) {
    const w = Math.max(2, Math.round(width || 256));
    const h = Math.max(STRIP_ROWS, Math.round(height || STRIP_ROWS));
    const rgba = readPixels(image, w, h);
    if (!rgba) {
      state.curves = null;
      state.chart = null;
      return;
    }
    // The ramp is a solid band, so one row out of its middle is the whole response.
    const row = Math.floor((RAMP_ROWS / STRIP_ROWS) * h * 0.5);
    const r = new Float32Array(w), g = new Float32Array(w), b = new Float32Array(w);
    for (let x = 0; x < w; x += 1) {
      const at = (row * w + x) * 4;
      r[x] = rgba[at] / 255;
      g[x] = rgba[at + 1] / 255;
      b[x] = rgba[at + 2] / 255;
    }
    state.curves = { r, g, b, width: w };
    state.chart = { image, width: w, height: h };
  }

  /**
   * Draw the whole panel.
   *
   * @returns {void}
   */
  function paint() {
    const ratio = surfaceRatio(element);
    const width = Math.max(1, Math.round(element.clientWidth));
    const height = Math.max(1, Math.round(element.clientHeight));
    if (canvas.width !== Math.round(width * ratio)) canvas.width = Math.round(width * ratio);
    if (canvas.height !== Math.round(height * ratio)) canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const theme = readTheme();
    let y = PAD;
    y = drawFacts(ctx, theme, width, y);
    if (!withChart) return;
    const plotBottom = height - PAD - CHART_HEIGHT - 6;
    drawCurves(ctx, theme, width, y, Math.max(0, plotBottom - y));
    drawChart(ctx, width, plotBottom + 6);
  }

  /**
   * Draw the summary line and the named facts.
   *
   * @param {CanvasRenderingContext2D} ctx - Where to draw.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {number} width - Panel width.
   * @param {number} top - Where to start drawing.
   * @returns {number} Where the next band starts.
   */
  function drawFacts(ctx, theme, width, top) {
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    let y = top;

    ctx.font = "11px sans-serif";
    ctx.fillStyle = theme.fg ?? "#cccccc";
    const identity = (state.facts ?? []).find(([name]) => name === "effect")?.[1];
    ctx.fillText(state.summary || "Not run", PAD, y + FACT_LINE / 2);
    y += FACT_LINE + 2;

    if (identity) {
      ctx.fillStyle = theme.warning ?? "#e0a33e";
      ctx.font = "10px sans-serif";
      ctx.fillText(String(identity), PAD, y + FACT_LINE / 2);
      y += FACT_LINE;
    }

    const pairs = (state.facts ?? []).filter(([name]) => name !== "effect");
    if (pairs.length) {
      ctx.font = "10px sans-serif";
      ctx.fillStyle = theme.descripText ?? "#999999";
      const line = pairs.map(([name, value]) => `${name} ${value}`).join("   ");
      ctx.fillText(line, PAD, y + FACT_LINE / 2, Math.max(0, width - PAD * 2));
      y += FACT_LINE;
    }
    return y + 4;
  }

  /**
   * Draw the transfer curves read off the strip.
   *
   * @param {CanvasRenderingContext2D} ctx - Where to draw.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {number} width - Panel width.
   * @param {number} top - Top of the plot.
   * @param {number} size - Height available to it.
   * @returns {void}
   */
  function drawCurves(ctx, theme, width, top, size) {
    if (!(size > 8)) return;
    const left = PAD;
    const plotWidth = Math.max(1, width - PAD * 2);

    ctx.fillStyle = theme.inputBg ?? "#1a1a1a";
    ctx.fillRect(left, top, plotWidth, size);
    ctx.strokeStyle = theme.border ?? "#333333";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(left, top + size);
    ctx.lineTo(left + plotWidth, top);
    ctx.stroke();
    ctx.setLineDash([]);

    const curves = state.curves;
    if (!curves) {
      ctx.fillStyle = theme.descripText ?? "#777777";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run node", left + plotWidth / 2, top + size / 2);
      ctx.textAlign = "left";
      return;
    }

    for (const [name, colour] of strokes) {
      const values = curves[name];
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      for (let i = 0; i < curves.width; i += 1) {
        const x = left + (i / (curves.width - 1)) * plotWidth;
        const y = top + size - Math.min(1, Math.max(0, values[i])) * size;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  /**
   * Draw the hue sweep and the memory colours, graded.
   *
   * @param {CanvasRenderingContext2D} ctx - Where to draw.
   * @param {number} width - Panel width.
   * @param {number} top - Top of the chart band.
   * @returns {void}
   */
  function drawChart(ctx, width, top) {
    const chart = state.chart;
    if (!chart?.image) return;
    const left = PAD;
    const plotWidth = Math.max(1, width - PAD * 2);
    // Only the bands below the ramp: the ramp itself is the plot above.
    const skip = Math.round((RAMP_ROWS / STRIP_ROWS) * chart.height);
    ctx.drawImage(
      chart.image,
      0, skip, chart.width, chart.height - skip,
      left, top, plotWidth, CHART_HEIGHT,
    );
  }

  try {
    releaseRatio = watchSurfaceRatio(element, schedulePaint);
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to watch the drawing resolution:`, error);
  }
  try {
    releaseRun = onRunEnded(() => { load(); });
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to watch for the end of a run:`, error);
  }
  // The panel is drawn into a canvas, which takes literal colours, so a palette change repaints.
  let releaseTheme = onThemeChange(schedulePaint);

  return {
    element,
    height: withChart ? UI_HEIGHT : FACTS_HEIGHT,
    maxHeight: withChart ? MAX_UI_HEIGHT : FACTS_HEIGHT,
    minWidth: MIN_UI_WIDTH,
    schedulePaint,
    load,
    dispose() {
      state.disposed = true;
      try {
        releaseRatio?.();
        releaseRun?.();
        releaseTheme?.();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to release the panel's watches:`, error);
      }
      releaseRatio = null;
      releaseRun = null;
      releaseTheme = null;
    },
  };
}

/**
 * Append the panel to a node.
 *
 * @param {object} node - The node being created.
 * @param {boolean} withChart - Whether this node gets the curves and the chart.
 * @returns {void}
 */
function attachLutPanel(node, withChart) {
  const panel = createLutPanel(node, withChart);
  appendInterfaceWidget(node, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      panel.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the panel:`, error);
    }
    return result;
  };

  panel.schedulePaint();
  panel.load();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "LUT", "Show what the table is"],
      name: "Show the LUT readout",
      tooltip:
        "Draw what a LUT node built on the node: its name, size and shape, whether it " +
        "changes anything at all, and on Load LUT and LUT Blender the transfer curves and " +
        "a graded reference chart. This applies to nodes added after the setting changes.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;
    const withChart = NODES_WITH_CHART.includes(nodeData?.name);

    const proto = nodeType.prototype;
    if (proto.__was_lut_panel_wrapped) return;
    proto.__was_lut_panel_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (panelEnabled()) attachLutPanel(this, withChart);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the LUT readout:`, error);
      }
      return result;
    };
  },
});
