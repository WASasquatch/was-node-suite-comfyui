/**
 * The tones Image Histogram Chart measured, drawn on the node itself.
 *
 * The curve is counted in the browser from the picture the node published, with
 * `histogram(rgba, 256)`. The black and white points come from the node.
 */

import { app } from "../../scripts/app.js";
import { ICON, ICON_SIZE, drawFidelityGlyph } from "./interface/icons.js";
import { histogram, readPixels } from "./interface/image_metrics.js";
import { PREVIEW_STATE, fetchInputPreview, watchPreviews } from "./interface/preview.js";
import { createReportPanel } from "./interface/report_panel.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onThemeChange, readTheme, themeVar } from "./interface/theme.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ImageAnalyzeUI";
const LOG_NAME = "WASNodeSuite.ImageAnalyze";
const SETTING_ID = "WAS.Analyze.ShowHistogram";
const NODE_ID = "Image Analyze";

const UI_WIDGET_NAME = "was_image_analyze_ui";
const UI_WIDGET_TYPE = "was_image_analyze";

// Height in node units the panel opens at: the summary, the tiles, the plot and the footer.
const PANEL_HEIGHT = 172;

// One bin per 8-bit level, which is what makes the count the same one PIL makes.
const BINS = 256;

// What each channel's curve is drawn in. The three tints `metrics_panel.js` uses, so two
// histograms in the pack read alike.
const TINTS = { r: "#e0564f", g: "#57b45c", b: "#5b8fe0" };

// What the glyph claims with no picture behind it, which is both the state before the first run
// and the state a cleared panel returns to.
const NOTHING_COUNTED = Object.freeze({
  icon: ICON.WARNING,
  detail: "no picture has arrived, so nothing is counted",
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
 * What the counted picture is worth as a measurement of what the node measured.
 *
 * @param {object|null} preview - The answer `fetchInputPreview` resolved to.
 * @returns {{icon: string, detail: string}} The glyph to draw and what it says on hover.
 */
export function histogramFidelity(preview) {
  const image = preview?.image;
  if (!image) return NOTHING_COUNTED;
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  const sourceWidth = Number(preview.sourceWidth) || width;
  const sourceHeight = Number(preview.sourceHeight) || height;
  if (sourceWidth === width && sourceHeight === height) {
    return {
      icon: ICON.EXACT,
      detail: `every one of the ${width}x${height} pixels the node charted, counted per level`,
    };
  }
  return {
    icon: ICON.APPROXIMATE,
    detail: `counted at ${width}x${height}, reduced from ${sourceWidth}x${sourceHeight} by `
      + "interface.preview_max_edge",
  };
}

/**
 * Build the histogram plot a report panel draws between its counts and its rows.
 *
 * @param {object} node - The node the plot belongs to, for its id and its redraws.
 * @returns {{element: HTMLElement, update: (report: object) => void, clear: () => void,
 *   dispose: () => void}} The band, in the shape `createReportPanel` takes as its sketch.
 */
export function createHistogramBand(node) {
  const element = document.createElement("div");
  element.style.cssText = "flex:1 1 auto;min-height:0;display:flex;flex-direction:column;gap:4px";

  const plot = document.createElement("canvas");
  plot.style.cssText = "flex:1 1 auto;min-height:0;width:100%;display:block;border-radius:2px";
  element.appendChild(plot);

  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;align-items:center;gap:6px;flex:0 0 auto;font-size:9px";
  element.appendChild(footer);

  const glyph = document.createElement("canvas");
  glyph.style.cssText = `width:${ICON_SIZE}px;height:${ICON_SIZE}px;flex:0 0 auto`;
  const note = document.createElement("span");
  note.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + `white-space:nowrap;color:${themeVar("fgMuted")}`;
  const range = document.createElement("span");
  range.style.cssText = "flex:0 0 auto";
  range.textContent = "0 - 255";
  footer.append(glyph, note, range);

  let disposed = false;
  let bins = null;
  let marks = null;
  let claim = NOTHING_COUNTED;
  let words = "";
  // Which run the picture on screen was counted from, so the report's second draw of the same
  // run does not fetch the whole PNG again.
  let counted = null;

  /** Draw the fidelity glyph the last count earned. */
  const drawGlyph = () => {
    drawFidelityGlyph(glyph, claim);
  };

  /** Draw the three curves and the two level markers over them. */
  const drawPlot = () => {
    if (disposed) return;
    const theme = readTheme();
    const width = plot.clientWidth;
    const height = plot.clientHeight;
    if (!(width > 0) || !(height > 0)) return;
    const ratio = surfaceRatio(plot);
    const w = Math.max(1, Math.round(width * ratio));
    const h = Math.max(1, Math.round(height * ratio));
    if (plot.width !== w || plot.height !== h) { plot.width = w; plot.height = h; }
    const ctx = plot.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = theme.bgDark;
    ctx.fillRect(0, 0, w, h);
    if (!bins) return;

    // One scale for all three, so a channel is read against the others rather than each being
    // stretched to the full height of its own peak.
    const peak = bins.peak || 1;
    const trace = (counts) => {
      ctx.beginPath();
      for (let i = 0; i < counts.length; i += 1) {
        const x = (i / (counts.length - 1)) * w;
        const y = h - (counts[i] / peak) * h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
    };
    for (const channel of ["r", "g", "b"]) {
      trace(bins[channel]);
      ctx.lineTo(w, h);
      ctx.lineTo(0, h);
      ctx.closePath();
      ctx.globalAlpha = 0.45;
      ctx.fillStyle = TINTS[channel];
      ctx.fill();
      ctx.globalAlpha = 1;
      // Outlined as well as filled: three translucent fills over each other muddy into one
      // shape, and the edge is what a channel's distribution is read from.
      trace(bins[channel]);
      ctx.strokeStyle = TINTS[channel];
      ctx.lineWidth = Math.max(1, ratio);
      ctx.stroke();
    }

    if (!marks) return;
    ctx.strokeStyle = theme.error;
    ctx.lineWidth = Math.max(1, ratio);
    ctx.setLineDash([3 * ratio, 3 * ratio]);
    for (const level of marks) {
      const x = Math.min(w - 0.5, Math.max(0.5, (level / 255) * w));
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  };

  /**
   * Count one picture and draw what came of it.
   *
   * @param {object|null} preview - The answer `fetchInputPreview` resolved to.
   * @returns {void}
   */
  const measure = (preview) => {
    if (disposed) return;
    bins = null;
    if (preview?.image) {
      // Read at the picture's own decoded size, so nothing resamples between the bytes the
      // route served and the counts drawn from them.
      const pixels = readPixels(
        preview.image, preview.image.naturalWidth, preview.image.naturalHeight,
      );
      if (pixels) bins = histogram(pixels, BINS);
    }
    claim = histogramFidelity(preview);
    // A run whose picture never arrived is not remembered, so the end of the run is a second
    // chance at it rather than a repeat of the same miss.
    if (!preview?.image) counted = null;
    words = bins
      ? (marks ? "dashed red: black and white points" : "red, green and blue counts")
      : (preview?.state === PREVIEW_STATE.WAITING
        ? "the picture has not been published yet"
        : preview?.label || "no picture to count");
    note.textContent = words;
    drawPlot();
    drawGlyph();
    node.setDirtyCanvas?.(true, false);
  };

  let fetching = false;
  let again = false;
  const load = async () => {
    if (disposed) return;
    // Asked again while a read is in flight, the picture already on its way is an older run's,
    // so the ask is remembered and served after it rather than dropped.
    if (fetching) {
      again = true;
      return;
    }
    fetching = true;
    try {
      do {
        again = false;
        measure(await fetchInputPreview(node));
      } while (again && !disposed);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the picture the node charted:`, error);
    } finally {
      fetching = false;
    }
  };

  /**
   * Read one report and count the picture the run published with it.
   *
   * @param {object|null} report - The report `createReportPanel` is drawing.
   * @returns {void}
   */
  const update = (report) => {
    if (disposed) return;
    const black = countOf(report, "black point");
    const white = countOf(report, "white point");
    marks = black === null || white === null ? null : [black, white];
    drawPlot();
    // The same report is drawn twice, once when the node finishes and once when the run ends,
    // and both draws describe the one picture. `run` counts every publish the process has made,
    // so two draws carrying the same number are two draws of one run.
    const run = Number(report?.run);
    if (Number.isFinite(run) && run === counted) return;
    counted = Number.isFinite(run) ? run : null;
    load();
  };

  const release = watchPreviews(node);

  const observer = typeof ResizeObserver === "function"
    ? new ResizeObserver(() => drawPlot())
    : null;
  observer?.observe(plot);
  const stopWatching = watchSurfaceRatio(plot, () => { drawPlot(); drawGlyph(); });
  // The plot and the glyph are canvases, which take literal colours, so a palette change draws
  // them again.
  const stopTheme = onThemeChange(() => { drawPlot(); drawGlyph(); });

  drawGlyph();

  return {
    element,
    update,
    clear() {
      bins = null;
      marks = null;
      counted = null;
      claim = NOTHING_COUNTED;
      note.textContent = "";
      drawPlot();
      drawGlyph();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      release();
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
      category: ["WAS Node Suite", "Analyze", "Show the histogram"],
      name: "Draw the histogram on Image Histogram Chart",
      tooltip:
        "Draw the red, green and blue counts of the measured image on the node, with the black "
        + "and white points marked in red. The node sends the browser a copy of the picture for "
        + "this, which is one PNG encode per run while the panel is open. The chart output is "
        + "unchanged either way.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_image_analyze_wrapped) return;
    proto.__was_image_analyze_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const node = this;
        const panel = createReportPanel(node, {
          className: "was-image-analyze",
          // The two facts the node publishes name the chart and what it was measured on, which
          // the summary and the glyph's hover already carry.
          facts: false,
          height: PANEL_HEIGHT,
          emptyLabel: "No tones measured yet",
          sketch: () => createHistogramBand(node),
          logName: LOG_NAME,
          failure: "Failed to read what the node charted:",
        });
        appendInterfaceWidget(node, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the histogram readout:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the histogram readout:`, error);
      }
      return result;
    };
  },
});
