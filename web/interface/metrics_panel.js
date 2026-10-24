/**
 * A band of image measurements, for any interface that holds a picture or two.
 *
 * `createMetricsSection` answers an element and an `update`. Hand it one picture and it reports
 * what that picture is; two and it adds how they differ.
 */

import { comparePair, describeOne, readPixels } from "./image_metrics.js";
import { createFigureTile } from "./report_panel.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onThemeChange, readTheme, themeVar } from "./theme.js";

const LOG_NAME = "WASNodeSuite.Metrics";

// Height of the histogram plot in CSS pixels.
const PLOT_HEIGHT = 46;

// The longest edge the measurements are taken at. A comparison does not change meaningfully
// above this, and SSIM over a 4K frame in a browser is seconds rather than milliseconds. A
// caller measuring on every widget change asks for less.
const WORKING_EDGE = 1024;

// What each figure is called and how it is written, in the order they are drawn.
const PAIR_FIGURES = [
  { key: "psnr", name: "PSNR", unit: " dB", digits: 2, hint: "higher is closer; over 40 dB is near identical" },
  { key: "ssim", name: "SSIM", unit: "", digits: 4, hint: "1.0 is identical structure" },
  { key: "rmse", name: "RMSE", unit: "", digits: 2, hint: "average error, 0 to 255" },
  { key: "mae", name: "MAE", unit: "", digits: 2, hint: "average absolute error, 0 to 255" },
  { key: "deltaE", name: "dE", unit: "", digits: 2, hint: "mean CIE76 colour shift; under 2 is hard to see" },
];

const SINGLE_FIGURES = [
  { key: "lumaMean", name: "MEAN", unit: "", digits: 1, hint: "average brightness, 0 to 255" },
  { key: "entropy", name: "ENTROPY", unit: " b", digits: 2, hint: "detail in the brightness distribution, 0 to 8 bits" },
  { key: "sharpness", name: "EDGES", unit: "", digits: 1, hint: "mean absolute laplacian; higher is more edge energy" },
];

/**
 * The size measurements are taken at, which bounds the work on a large frame.
 *
 * @param {number} width - The picture's width.
 * @param {number} height - Its height.
 * @param {number} edge - The longest edge to measure at.
 * @returns {{width: number, height: number, scaled: boolean}} The working size. A size that is
 *   not a number is left alone rather than scaled by a factor that is not one either.
 */
export function workingSize(width, height, edge) {
  const longest = Math.max(width, height);
  if (!(longest > edge)) return { width, height, scaled: false };
  const factor = edge / longest;
  return {
    width: Math.max(1, Math.round(width * factor)),
    height: Math.max(1, Math.round(height * factor)),
    scaled: true,
  };
}

/**
 * Build a band of measurements for one picture or a pair.
 *
 * @param {object} [options] - How it is drawn.
 * @param {boolean} [options.histogram] - Draw the histogram plot. On by default.
 * @param {string} [options.mismatch] - What a pair of different sizes means. `resample`, the
 *   default, reads both at the first picture's working size and says so. `refuse` reports the
 *   first picture alone and names both sizes, for a band whose figures are only meaningful
 *   pixel against pixel.
 * @param {number} [options.workingEdge] - The longest edge the measurements are taken at,
 *   1024 by default. Halving it is four times less work per update.
 * @returns {{element: HTMLElement, update: Function, clear: Function}} The band, an `update`
 *   taking one or two decoded pictures, and a `clear`.
 */
export function createMetricsSection(options = {}) {
  const withPlot = options.histogram !== false;
  const refuseMismatch = options.mismatch === "refuse";
  const edge = Number(options.workingEdge) > 0 ? Number(options.workingEdge) : WORKING_EDGE;

  const root = document.createElement("div");
  root.style.cssText = [
    "display:flex", "flex-direction:column", "gap:4px",
    "flex:1 1 auto", "min-height:0", "overflow:hidden",
    "font:10px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
  ].join(";");

  const figures = document.createElement("div");
  figures.style.cssText = "display:flex;flex-wrap:wrap;gap:10px 16px;flex:0 0 auto";
  root.appendChild(figures);

  const plot = document.createElement("canvas");
  plot.style.cssText = `width:100%;height:${PLOT_HEIGHT}px;display:${withPlot ? "block" : "none"};`
    + "flex:0 0 auto;border-radius:2px";
  root.appendChild(plot);

  const legend = document.createElement("div");
  legend.style.cssText = `display:${withPlot ? "flex" : "none"};justify-content:space-between;`
    + "font-size:9px;flex:0 0 auto";
  root.appendChild(legend);

  let last = null;

  /**
   * Draw one figure.
   *
   * @param {object} spec - Its name, unit, digits and hint.
   * @param {number} value - What to show.
   * @returns {void}
   */
  const addFigure = (spec, value) => {
    const written = Number.isFinite(value)
      ? `${value.toFixed(spec.digits)}${spec.unit}`
      : "identical";
    figures.appendChild(createFigureTile(spec.name, written, spec.hint));
  };

  /** Draw the histogram, one picture filled and a second as a line over it. */
  const drawPlot = () => {
    if (!withPlot || !last) return;
    const theme = readTheme();
    // The graph's zoom is a transform on an ancestor: the plot's layout size never moves, so a
    // backing store sized from the device ratio alone is stretched by the GPU and the histogram
    // goes soft on zoom in. `surfaceRatio` folds the zoom in, and `watchSurfaceRatio` is what
    // reports the change, since a transform fires no event and reaches no ResizeObserver.
    const ratio = surfaceRatio(plot);
    // Layout pixels, not the client rectangle, which the zoom has already multiplied.
    const width = plot.clientWidth;
    if (!(width > 0)) return;
    const w = Math.max(1, Math.round(width * ratio));
    const h = Math.max(1, Math.round(PLOT_HEIGHT * ratio));
    if (plot.width !== w || plot.height !== h) { plot.width = w; plot.height = h; }
    const ctx = plot.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = theme.panelBg;
    ctx.fillRect(0, 0, w, h);

    const tint = { r: "#e0564f", g: "#57b45c", b: "#5b8fe0" };
    // One scale for both, or the shorter picture's curve is stretched to full height and the
    // two cannot be read against each other, which is the only reason to draw them together.
    const peak = Math.max(last.a.histogram.peak || 1, last.b?.histogram.peak || 0);

    const trace = (bins) => {
      ctx.beginPath();
      for (let i = 0; i < bins.length; i += 1) {
        const x = (i / (bins.length - 1)) * w;
        const y = h - (bins[i] / peak) * h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
    };
    const draw = (hist, style) => {
      for (const channel of ["r", "g", "b"]) {
        const bins = hist[channel];
        if (style === "fill") {
          trace(bins);
          ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
          ctx.globalAlpha = 0.5;
          ctx.fillStyle = tint[channel];
          ctx.fill();
          ctx.globalAlpha = 1;
          // Outlined as well as filled: three translucent fills over each other muddy into one
          // shape, and the edge is what a channel's distribution is actually read from.
          trace(bins);
          ctx.strokeStyle = tint[channel];
          ctx.lineWidth = Math.max(1, ratio);
          ctx.stroke();
        } else {
          trace(bins);
          ctx.strokeStyle = tint[channel];
          ctx.lineWidth = Math.max(1, ratio);
          ctx.setLineDash([3 * ratio, 3 * ratio]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    };
    draw(last.a.histogram, "fill");
    // Drawn only where it would differ. On an identical pair the dashes land exactly on A's
    // outline and hide the very thing they are drawn over.
    if (last.b && !last.identical) draw(last.b.histogram, "line");
  };

  /**
   * Measure and draw.
   *
   * @param {CanvasImageSource|null} imageA - The first picture.
   * @param {CanvasImageSource|null} [imageB] - The second, for a comparison.
   * @param {object} [sizes] - `{aWidth, aHeight, bWidth, bHeight}` at source resolution, for
   *   reporting a mismatch. Left out, the pictures' own sizes are used.
   * @returns {void}
   */
  const update = (imageA, imageB = null, sizes = {}) => {
    figures.textContent = "";
    legend.textContent = "";
    if (!imageA) { last = null; drawPlot(); return; }
    try {
      const aw = sizes.aWidth || imageA.naturalWidth || imageA.width;
      const ah = sizes.aHeight || imageA.naturalHeight || imageA.height;
      const bw = sizes.bWidth || imageB?.naturalWidth;
      const bh = sizes.bHeight || imageB?.naturalHeight;
      const mismatch = Boolean(imageB && bw && bh && (bw !== aw || bh !== ah));
      // A figure over a resampled B is a figure about the resampling as much as about the
      // pictures, so a band that says it compares pixel to pixel reports A alone instead.
      const refused = mismatch && refuseMismatch;
      const work = workingSize(aw, ah, edge);
      const pixelsA = readPixels(imageA, work.width, work.height);
      if (!pixelsA) { last = null; return; }
      const a = describeOne(pixelsA, work.width, work.height);
      let b = null;
      let pair = null;
      if (imageB && !refused) {
        // Both read at the same size, so a comparison of two different resolutions is a
        // comparison of the same picture area rather than a refusal.
        const pixelsB = readPixels(imageB, work.width, work.height);
        if (pixelsB) {
          b = describeOne(pixelsB, work.width, work.height);
          pair = comparePair(pixelsA, pixelsB, work.width, work.height);
        }
      }
      // Whether the two are the same picture, which changes what is worth drawing and saying.
      const identical = Boolean(pair) && pair.rmse === 0;
      last = { a, b, identical };

      if (pair) for (const spec of PAIR_FIGURES) addFigure(spec, pair[spec.key]);
      else for (const spec of SINGLE_FIGURES) addFigure(spec, a[spec.key]);

      const note = document.createElement("span");
      note.style.color = themeVar("fgMuted");
      note.textContent = [
        withPlot ? (b && !last.identical ? "solid A, dashed B"
          : b ? "A and B identical" : "histogram") : "",
        work.scaled ? `measured at ${work.width}x${work.height}` : "",
        refused ? `A only: A is ${aw}x${ah} and B is ${bw}x${bh}`
          : mismatch ? `B resampled from ${bw}x${bh}` : "",
      ].filter(Boolean).join("   ");
      const scale = document.createElement("span");
      scale.style.color = themeVar("fgMuted");
      scale.textContent = "0 - 255";
      legend.append(note, scale);
      drawPlot();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to measure:`, error);
      last = null;
    }
  };

  // Two things change the plot: the node being resized, which only a size observer sees, and
  // the graph's zoom, which only the ratio watcher sees.
  const observer = typeof ResizeObserver === "function"
    ? new ResizeObserver(() => drawPlot())
    : null;
  observer?.observe(plot);
  const stopWatching = watchSurfaceRatio(plot, () => drawPlot());
  // The plot is a canvas, which takes literal colours, so a palette change redraws it.
  const stopTheme = onThemeChange(() => drawPlot());
  // A canvas the browser gave back after losing it comes back blank.
  plot.addEventListener("contextrestored", () => drawPlot());

  return {
    element: root,
    update,
    clear() { last = null; figures.textContent = ""; legend.textContent = ""; drawPlot(); },
    dispose() { observer?.disconnect(); stopWatching(); stopTheme(); },
  };
}
