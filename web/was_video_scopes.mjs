/**
 * Video Scopes' panel: the clip playing beside its waveform, vectorscope, histogram, RGB parade
 * and a timeline, each drawn live from the frame data the node measured.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { dressPanel } from "./interface/card.mjs";
import { addressOf, hold, keptEntry, release } from "./interface/compare_media.mjs";
import { executionId, nodeLocator, ownOutput } from "./interface/preview.mjs";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.mjs";
import { onRunEnded } from "./interface/run_events.mjs";
import { themeVar } from "./interface/theme.mjs";
import { appendInterfaceWidget, graphLoading, keepOnNode } from "./interface/widget.mjs";

const EXT_NAME = "WASNodeSuite.VideoScopes";
const NODE_ID = "WASVideoScopes";
const UI_WIDGET_NAME = "was_video_scopes_ui";
const UI_WIDGET_TYPE = "was_video_scopes";

// Height the panel is drawn at before the node is resized, and the narrowest worth drawing.
const PANEL_HEIGHT = 460;
const PANEL_MIN_WIDTH = 420;

// Where the last files written are kept on the node; no python reads it.
const REMEMBERED_KEY = "was_video_scopes";

// The files the node writes, by the `ui` key each arrives under.
const PARTS = { source: "scope_source", data: "scope_data", spectrum: "scope_spectrum" };

// Widest a playing frame is read at for the scopes, in pixels.
const SAMPLE_WIDTH = 640;

// BT.709 luma weights and the Cb, Cr divisors.
const LUMA = [0.2126, 0.7152, 0.0722];
const CB_DIVISOR = 1.8556;
const CR_DIVISOR = 1.5748;

const PAGE = [17, 18, 17];
const PANEL = [25, 27, 25];
const EDGE = "rgb(74, 96, 72)";
const ACCENT = [163, 210, 155];
const TRACES = [[255, 92, 92], [110, 226, 120], [96, 150, 255], [226, 232, 224]];
const CURVES = [["luma", "rgb(232,236,230)"], ["highlights", "rgb(255,110,110)"],
  ["shadows", "rgb(150,120,255)"], ["saturation", "rgb(163,210,155)"]];
const SPECTRUM_STOPS = [[0, [10, 11, 10]], [0.35, [28, 70, 40]], [0.7, ACCENT], [1, [250, 252, 248]]];

const rgb = (c) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

/** The panel's boxes for a canvas of this size, as `[x, y, w, h]`. */
function layout(width, height) {
  const m = Math.round(height * 0.018);
  const g = Math.round(height * 0.014);
  const topH = Math.round(height * 0.5);
  const midH = Math.round(height * 0.22);
  const y1 = m + topH + g;
  const y2 = y1 + midH + g;
  const side = Math.min(topH, Math.round((width - 2 * m - 2 * g) * 0.3));
  const sourceW = Math.round((width - 2 * m - 2 * g - side) * 0.6);
  const waveW = width - 2 * m - 2 * g - sourceW - side;
  const histW = Math.round((width - 2 * m - g) * 0.32);
  return {
    source: [m, m, sourceW, topH],
    waveform: [m + sourceW + g, m, waveW, topH],
    vectorscope: [width - m - side, m, side, topH],
    histogram: [m, y1, histW, midH],
    parade: [m + histW + g, y1, width - 2 * m - histW - g, midH],
    timeline: [m, y2, width - 2 * m, height - m - y2],
  };
}


/** A 256-entry colour table for the spectrogram, black through green to white. */
function spectrumColours() {
  const table = new Uint8ClampedArray(256 * 3);
  for (let index = 0; index < 256; index += 1) {
    const t = index / 255;
    for (let s = 0; s < SPECTRUM_STOPS.length - 1; s += 1) {
      const [t0, c0] = SPECTRUM_STOPS[s];
      const [t1, c1] = SPECTRUM_STOPS[s + 1];
      if (t <= t1) {
        const share = (t - t0) / Math.max(t1 - t0, 1e-6);
        for (let k = 0; k < 3; k += 1) table[index * 3 + k] = c0[k] + (c1[k] - c0[k]) * share;
        break;
      }
    }
  }
  return table;
}

/**
 * Build the scopes panel for one node.
 *
 * @param {object} node - The node the panel belongs to.
 * @returns {object} The element, its height, its minimum width, `dispose` and `refresh`.
 */
function createScopesPanel(node) {
  const root = document.createElement("div");
  root.className = "was-video-scopes";
  Object.assign(root.style, {
    position: "relative", width: "100%", height: "100%", display: "flex", flexDirection: "column",
    gap: "4px", padding: "4px",
  });
  dressPanel(root);

  const stage = document.createElement("div");
  Object.assign(stage.style, { position: "relative", flex: "1 1 auto", minHeight: "0", overflow: "hidden" });
  const video = document.createElement("video");
  video.loop = true;
  video.playsInline = true;
  video.preload = "auto";
  Object.assign(video.style, {
    position: "absolute", inset: "0", width: "100%", height: "100%", opacity: "0", pointerEvents: "none",
  });
  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, { position: "absolute", inset: "0", width: "100%", height: "100%", display: "block" });
  const context = canvas.getContext("2d");
  const empty = document.createElement("div");
  empty.textContent = "run to measure";
  Object.assign(empty.style, {
    position: "absolute", inset: "0", display: "flex", alignItems: "center", justifyContent: "center",
    color: themeVar("fgMuted"), fontSize: "11px",
  });
  stage.append(video, canvas, empty);

  const bar = document.createElement("div");
  Object.assign(bar.style, { display: "flex", alignItems: "center", gap: "8px", flex: "0 0 auto", fontSize: "11px" });
  const toggle = document.createElement("button");
  toggle.textContent = "play";
  Object.assign(toggle.style, {
    fontSize: "11px", padding: "2px 12px", cursor: "pointer", borderRadius: "999px",
    background: themeVar("panelBg"), color: themeVar("fg"), border: `1px solid ${themeVar("border")}`,
  });
  const readout = document.createElement("span");
  Object.assign(readout.style, { flex: "1 1 auto", minWidth: "0", color: themeVar("fgMuted"), fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" });
  bar.append(toggle, readout);
  root.append(stage, bar);

  const state = { doc: null, images: {}, loaded: "", addresses: [], turn: 0, timeline: null, timelineKey: "" };
  const sampler = document.createElement("canvas");
  const samplerPen = sampler.getContext("2d", { willReadFrequently: true });
  const scopes = { key: "", wave: null, parade: null, vector: null, hist: null };
  const colours = spectrumColours();

  const frameNow = () => {
    const doc = state.doc;
    if (!doc) return 0;
    return Math.max(0, Math.min(doc.frames - 1, Math.floor(video.currentTime * doc.rate + 1e-4)));
  };

  // The playing frame's pixels, read at most `SAMPLE_WIDTH` wide.
  const sampleFrame = () => {
    if (!(video.readyState >= 2 && video.videoWidth > 0)) return null;
    const w = Math.min(SAMPLE_WIDTH, video.videoWidth);
    const h = Math.max(1, Math.round((w * video.videoHeight) / video.videoWidth));
    if (sampler.width !== w || sampler.height !== h) { sampler.width = w; sampler.height = h; }
    samplerPen.drawImage(video, 0, 0, w, h);
    return { w, h, data: samplerPen.getImageData(0, 0, w, h).data };
  };

  // Counts painted from the panel colour to a trace colour by their logarithm, the busiest cell brightest.
  const paint = (counts, cols, rows, colour, target) => {
    let peak = 0;
    for (let i = 0; i < counts.length; i += 1) if (counts[i] > peak) peak = counts[i];
    const canvasOut = target ?? document.createElement("canvas");
    canvasOut.width = cols;
    canvasOut.height = rows;
    const pen = canvasOut.getContext("2d");
    const out = pen.createImageData(cols, rows);
    const scale = peak > 0 ? 1 / Math.log1p(peak) : 0;
    for (let i = 0; i < counts.length; i += 1) {
      const v = Math.min(1, Math.log1p(counts[i]) * scale);
      const o = i * 4;
      out.data[o] = PANEL[0] + (colour[0] - PANEL[0]) * v;
      out.data[o + 1] = PANEL[1] + (colour[1] - PANEL[1]) * v;
      out.data[o + 2] = PANEL[2] + (colour[2] - PANEL[2]) * v;
      out.data[o + 3] = 255;
    }
    pen.putImageData(out, 0, 0);
    return canvasOut;
  };

  // A plane's waveform at the panel's own pixel size: channel 0 to 2, or 3 for luma.
  const waveform = (sample, channel, cols, rows) => {
    const { w, h, data } = sample;
    const counts = new Uint32Array(cols * rows);
    const top = rows - 1;
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const i = (y * w + x) * 4;
        const v = channel === 3 ? LUMA[0] * data[i] + LUMA[1] * data[i + 1] + LUMA[2] * data[i + 2] : data[i + channel];
        counts[(top - Math.round((v / 255) * top)) * cols + ((x * cols / w) | 0)] += 1;
      }
    }
    return counts;
  };

  // Where each pixel's Cb, Cr lands on a square of `side` pixels.
  const vectorscope = (sample, side, chroma) => {
    const { w, h, data } = sample;
    const counts = new Uint32Array(side * side);
    for (let p = 0; p < w * h; p += 1) {
      const i = p * 4;
      const luma = LUMA[0] * data[i] + LUMA[1] * data[i + 1] + LUMA[2] * data[i + 2];
      const cb = (data[i + 2] - luma) / CB_DIVISOR / 255;
      const cr = (data[i] - luma) / CR_DIVISOR / 255;
      const x = Math.floor(side * (0.5 + cb * chroma));
      const y = Math.floor(side * (0.5 - cr * chroma));
      if (x >= 0 && x < side && y >= 0 && y < side) counts[y * side + x] += 1;
    }
    return counts;
  };

  // 256-bin histograms of red, green, blue and luma, each scaled to its peak by square root.
  const histograms = (sample) => {
    const { w, h, data } = sample;
    const counts = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
    for (let p = 0; p < w * h; p += 1) {
      const i = p * 4;
      counts[0][data[i]] += 1;
      counts[1][data[i + 1]] += 1;
      counts[2][data[i + 2]] += 1;
      counts[3][Math.round(LUMA[0] * data[i] + LUMA[1] * data[i + 1] + LUMA[2] * data[i + 2])] += 1;
    }
    return counts.map((plane) => {
      let peak = 1;
      for (const value of plane) if (value > peak) peak = value;
      return Array.from(plane, (value) => Math.sqrt(value / peak));
    });
  };

  // Every scope of the playing frame at these sizes, worked out again only when the frame or a size changes.
  const analyse = (sizes) => {
    const key = `${video.currentTime.toFixed(4)}|${sizes.join(",")}|${state.loaded}`;
    if (scopes.key === key) return scopes;
    const sample = sampleFrame();
    if (!sample) return null;
    const [waveCols, waveRows, paradeCols, paradeRows, side] = sizes;
    scopes.wave = paint(waveform(sample, 3, waveCols, waveRows), waveCols, waveRows, ACCENT, scopes.wave);
    scopes.parade = [0, 1, 2].map((channel) => paint(
      waveform(sample, channel, paradeCols, paradeRows), paradeCols, paradeRows, TRACES[channel], scopes.parade?.[channel]));
    scopes.vector = paint(vectorscope(sample, side, state.doc?.chroma_scale ?? 0.7667), side, side, [209, 255, 209], scopes.vector);
    scopes.hist = histograms(sample);
    scopes.key = key;
    return scopes;
  };

  const panelBox = (box, title, scale) => {
    const [x, y, w, h] = box;
    context.fillStyle = rgb(PANEL);
    context.strokeStyle = EDGE;
    context.lineWidth = Math.max(1, scale);
    context.beginPath();
    context.roundRect?.(x + 0.5, y + 0.5, w - 1, h - 1, 5 * scale);
    context.fill();
    context.stroke();
    context.fillStyle = "rgb(222,228,220)";
    context.font = `${Math.round(11 * scale)}px sans-serif`;
    context.textBaseline = "top";
    context.fillText(title, x + 7 * scale, y + 4 * scale);
    const pad = Math.round(5 * scale);
    const top = y + Math.round(18 * scale);
    return [x + pad, top, w - 2 * pad, y + h - pad - top];
  };

  const timelinePicture = (w, h) => {
    const doc = state.doc;
    const key = `${w}x${h}|${state.loaded}`;
    if (state.timelineKey === key && state.timeline) return state.timeline;
    const picture = document.createElement("canvas");
    picture.width = Math.max(1, w);
    picture.height = Math.max(1, h);
    const pen = picture.getContext("2d");
    const sound = Boolean(state.images.spectrum && doc.spectrum);
    const curvesH = sound ? Math.round(h * 0.55) : h;
    const span = Math.max(1, doc.frames - 1);
    for (const [name, colour] of CURVES) {
      const values = doc.curves?.[name] || [];
      if (!values.length) continue;
      const scale = name === "luma" || name === "saturation" ? 1 : Math.max(0.02, Math.max(...values));
      pen.strokeStyle = colour;
      pen.lineWidth = 1;
      pen.beginPath();
      values.forEach((value, index) => {
        const px = (index * (w - 1)) / span;
        const py = (curvesH - 2) * (1 - Math.min(1, value / scale)) + 1;
        if (index === 0) pen.moveTo(px, py);
        else pen.lineTo(px, py);
      });
      pen.stroke();
    }
    if (sound) {
      const stripH = h - curvesH - 2;
      pen.imageSmoothingEnabled = true;
      pen.drawImage(state.images.spectrum, 0, curvesH + 2, w, stripH);
      const strip = pen.getImageData(0, curvesH + 2, w, stripH);
      for (let i = 0; i < strip.data.length; i += 4) {
        const v = strip.data[i];
        strip.data[i] = colours[v * 3];
        strip.data[i + 1] = colours[v * 3 + 1];
        strip.data[i + 2] = colours[v * 3 + 2];
      }
      pen.putImageData(strip, 0, curvesH + 2);
    }
    state.timeline = picture;
    state.timelineKey = key;
    return picture;
  };

  const drawGraticule = (x, y, side, scale) => {
    context.strokeStyle = "rgb(78,78,78)";
    context.lineWidth = Math.max(1, scale);
    context.beginPath();
    context.arc(x + side / 2, y + side / 2, (side / 2) * 0.92, 0, Math.PI * 2);
    context.moveTo(x + side / 2, y + side * 0.06);
    context.lineTo(x + side / 2, y + side * 0.94);
    context.moveTo(x + side * 0.06, y + side / 2);
    context.lineTo(x + side * 0.94, y + side / 2);
    context.stroke();
    // The colour targets: a box at 75%, a ring and its label at 100%.
    context.strokeStyle = "rgb(214,170,64)";
    context.fillStyle = "rgb(232,232,232)";
    context.font = `${Math.round(9 * scale)}px sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    for (const [label, level, tx, ty] of state.doc?.targets || []) {
      const cx = x + tx * side;
      const cy = y + ty * side;
      context.beginPath();
      if (level < 1) context.rect(cx - side * 0.018, cy - side * 0.018, side * 0.036, side * 0.036);
      else context.arc(cx, cy, side * 0.011, 0, Math.PI * 2);
      context.stroke();
      if (level >= 1) context.fillText(label, cx + (cx - (x + side / 2)) * 0.12, cy + (cy - (y + side / 2)) * 0.12);
    }
    context.textAlign = "start";
    context.textBaseline = "top";
    // The skin tone line, 123 degrees counter-clockwise from +Cb.
    const angle = (123 * Math.PI) / 180;
    context.strokeStyle = "rgb(232,150,112)";
    context.beginPath();
    context.moveTo(x + side / 2, y + side / 2);
    context.lineTo(x + side / 2 + Math.cos(angle) * side * 0.46, y + side / 2 - Math.sin(angle) * side * 0.46);
    context.stroke();
  };

  const draw = () => {
    if (!context) return;
    const ratio = surfaceRatio(canvas);
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    context.fillStyle = rgb(PAGE);
    context.fillRect(0, 0, width, height);
    const doc = state.doc;
    if (!doc) return;
    const scale = Math.max(0.6, height / 460);
    const boxes = layout(width, height);
    const frame = frameNow();

    const [sx, sy, sw, sh] = panelBox(boxes.source, "Source", scale);
    context.fillStyle = "#000";
    context.fillRect(sx, sy, sw, sh);
    if (video.readyState >= 2 && video.videoWidth > 0) {
      const fit = Math.min(sw / video.videoWidth, sh / video.videoHeight);
      const fw = video.videoWidth * fit;
      const fh = video.videoHeight * fit;
      context.drawImage(video, sx + (sw - fw) / 2, sy + (sh - fh) / 2, fw, fh);
    }

    const [wx, wy, wW, wH] = panelBox(boxes.waveform, "Luma waveform", scale);
    const [px, py, pW, pH] = panelBox(boxes.parade, "RGB parade", scale);
    const [vx, vy, vw, vh] = panelBox(boxes.vectorscope, "Vectorscope", scale);
    const side = Math.max(16, Math.floor(Math.min(vw, vh)));
    const third = Math.max(8, Math.floor(pW / 3));
    const live = analyse([Math.max(16, Math.round(wW)), Math.max(16, Math.round(wH)), third, Math.max(16, Math.round(pH)), side]);
    context.imageSmoothingEnabled = false;
    if (live) {
      context.drawImage(live.wave, Math.round(wx), Math.round(wy));
      live.parade.forEach((plane, channel) => context.drawImage(plane, Math.round(px + channel * third), Math.round(py)));
    }
    context.imageSmoothingEnabled = true;
    context.strokeStyle = "rgba(255,255,255,0.08)";
    for (const [bx, by, bw, bh] of [[wx, wy, wW, wH], [px, py, pW, pH]]) {
      for (const level of [0, 0.25, 0.5, 0.75, 1]) {
        const row = Math.round(by + (1 - level) * (bh - 1)) + 0.5;
        context.beginPath();
        context.moveTo(bx, row);
        context.lineTo(bx + bw, row);
        context.stroke();
      }
    }

    const ox = Math.round(vx + (vw - side) / 2);
    const oy = Math.round(vy + (vh - side) / 2);
    if (live) {
      context.imageSmoothingEnabled = false;
      context.drawImage(live.vector, ox, oy);
      context.imageSmoothingEnabled = true;
    }
    drawGraticule(ox, oy, side, scale);

    const [hx, hy, hw, hh] = panelBox(boxes.histogram, "Histogram", scale);
    if (live) {
      const bins = 256;
      context.globalCompositeOperation = "lighter";
      for (let channel = 0; channel < 3; channel += 1) {
        const c = TRACES[channel];
        const heights = live.hist[channel];
        context.fillStyle = `rgba(${c[0]}, ${c[1]}, ${c[2]}, 0.42)`;
        context.beginPath();
        context.moveTo(hx, hy + hh);
        for (let b = 0; b < bins; b += 1) context.lineTo(hx + (b / (bins - 1)) * hw, hy + hh - heights[b] * (hh - 2));
        context.lineTo(hx + hw, hy + hh);
        context.closePath();
        context.fill();
      }
      context.globalCompositeOperation = "source-over";
      context.strokeStyle = rgb(TRACES[3]);
      context.lineWidth = Math.max(1, scale);
      context.beginPath();
      for (let b = 0; b < bins; b += 1) {
        const lx = hx + (b / (bins - 1)) * hw;
        const ly = hy + hh - live.hist[3][b] * (hh - 2);
        if (b === 0) context.moveTo(lx, ly); else context.lineTo(lx, ly);
      }
      context.stroke();
    }

    const [tx, ty, tw, th] = panelBox(boxes.timeline, "Timeline", scale);
    context.font = `${Math.round(9.5 * scale)}px sans-serif`;
    let lx = boxes.timeline[0] + 70 * scale;
    for (const [name, colour] of CURVES) {
      const word = name === "highlights" ? "highlights clipped" : name === "shadows" ? "shadows crushed" : name;
      context.fillStyle = colour;
      context.fillText(word, lx, boxes.timeline[1] + 5 * scale);
      lx += context.measureText(word).width + 12 * scale;
    }
    context.drawImage(timelinePicture(Math.round(tw), Math.round(th)), tx, ty, tw, th);
    const column = tx + (frame * (tw - 1)) / Math.max(1, doc.frames - 1);
    context.fillStyle = rgb(ACCENT);
    context.fillRect(Math.round(column) - Math.max(1, scale), ty, Math.max(2, 2 * scale), th);
    state.timelineBox = [tx, ty, tw, th];

    const curves = doc.curves || {};
    const share = (name) => `${((curves[name]?.[frame] ?? 0) * 100).toFixed(1)}%`;
    readout.textContent = `frame ${frame + 1} / ${doc.frames} · ${video.currentTime.toFixed(2)} s · `
      + `luma ${share("luma")} · clipped ${share("highlights")} · crushed ${share("shadows")} · saturation ${share("saturation")}`;
  };

  let pending = 0;
  const tick = () => {
    pending = 0;
    draw();
    if (!video.paused && !video.ended) pending = requestAnimationFrame(tick);
  };
  const schedule = () => { if (!pending) pending = requestAnimationFrame(tick); };
  for (const name of ["play", "seeked", "loadeddata", "timeupdate"]) video.addEventListener(name, schedule);
  const resized = new ResizeObserver(() => schedule());
  resized.observe(stage);
  const stopRatio = watchSurfaceRatio(canvas, schedule);

  toggle.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!video.currentSrc) return;
    if (video.paused) video.play().catch(() => {});
    else video.pause();
    toggle.textContent = video.paused ? "play" : "pause";
  });
  video.addEventListener("pause", () => { toggle.textContent = "play"; });
  video.addEventListener("play", () => { toggle.textContent = "pause"; });

  // Seeking from the timeline, in the bubble phase so the widget root's capture does not take it.
  const seekFrom = (event) => {
    const box = state.timelineBox;
    const doc = state.doc;
    if (!box || !doc || !video.currentSrc) return false;
    const rect = canvas.getBoundingClientRect();
    const ratio = canvas.width / Math.max(1, rect.width);
    const x = (event.clientX - rect.left) * ratio;
    const y = (event.clientY - rect.top) * ratio;
    if (y < box[1] || y > box[1] + box[3]) return false;
    const share = Math.max(0, Math.min(1, (x - box[0]) / Math.max(1, box[2])));
    video.currentTime = (share * (doc.frames - 1) + 0.5) / doc.rate;
    schedule();
    return true;
  };
  const seeking = { on: false };
  canvas.addEventListener("pointerdown", (event) => {
    if (!seekFrom(event)) return;
    seeking.on = true;
    canvas.setPointerCapture?.(event.pointerId);
    event.stopPropagation();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!seeking.on) return;
    seekFrom(event);
    event.stopPropagation();
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!seeking.on) return;
    seeking.on = false;
    canvas.releasePointerCapture?.(event.pointerId);
    event.stopPropagation();
  });

  const letGo = () => {
    for (const address of state.addresses) release(address);
    state.addresses = [];
  };

  const entriesOf = (output) => {
    const found = {};
    for (const [part, key] of Object.entries(PARTS)) found[part] = output?.[key]?.[0] ?? null;
    const named = Object.values(found).filter(Boolean);
    return named.length && named.every(ownOutput) ? found : {};
  };

  const loadImage = (url) => new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => resolve(null);
    image.src = url;
  });

  /** Point the panel at one run's files; false when they name nothing to show. */
  function show(found) {
    const needed = ["source", "data"];
    if (!needed.every((part) => found?.[part])) return false;
    const addresses = Object.fromEntries(Object.entries(found).map(([part, entry]) => [part, addressOf(entry)]));
    const key = Object.values(addresses).join("|");
    if (key === state.loaded) return true;
    state.loaded = key;
    const mine = ++state.turn;
    const parts = Object.keys(addresses).filter((part) => addresses[part]);
    Promise.all(parts.map((part) => hold(addresses[part]))).then(async (urls) => {
      if (mine !== state.turn) {
        parts.forEach((part, index) => { if (urls[index]) release(addresses[part]); });
        return;
      }
      letGo();
      state.addresses = parts.filter((part, index) => urls[index]).map((part) => addresses[part]);
      const byPart = Object.fromEntries(parts.map((part, index) => [part, urls[index]]));
      if (!byPart.source || !byPart.data) { state.loaded = ""; return; }
      try {
        const doc = await (await fetch(byPart.data)).json();
        const images = {};
        if (byPart.spectrum) images.spectrum = await loadImage(byPart.spectrum);
        if (mine !== state.turn) return;
        state.doc = doc;
        state.images = images;
        state.timeline = null;
        video.src = byPart.source;
        empty.style.display = "none";
        schedule();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to read the measured clip:`, error);
        state.loaded = "";
      }
    });
    return true;
  }

  function remember(found) {
    if (JSON.stringify(node.properties?.[REMEMBERED_KEY]) === JSON.stringify(found)) return;
    keepOnNode(node, REMEMBERED_KEY, found);
  }

  function refresh() {
    try {
      const store = app?.nodeOutputs ?? {};
      const output = store[nodeLocator(node)] ?? store[executionId(node)] ?? store[String(node.id)];
      const found = entriesOf(output);
      if (show(found)) { remember(found); return; }
      const kept = node.properties?.[REMEMBERED_KEY];
      if (kept && Object.values(kept).filter(Boolean).every(keptEntry)) show(kept);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the node's files:`, error);
    }
  }

  const onExecuted = (event) => {
    try {
      const detail = event?.detail;
      const shown = String(detail?.display_node ?? detail?.node ?? "");
      if (!shown || shown !== executionId(node)) return;
      const found = entriesOf(detail?.output);
      if (show(found)) remember(found);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the node's report:`, error);
    }
  };
  api.addEventListener("executed", onExecuted);

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const configured = originalOnConfigure?.apply(this, args);
    if (!graphLoading()) refresh();
    return configured;
  };
  const stop = onRunEnded(() => refresh());
  refresh();

  const dispose = () => {
    try {
      stop?.();
      api.removeEventListener?.("executed", onExecuted);
      state.turn += 1;
      stopRatio();
      resized.disconnect();
      if (pending) cancelAnimationFrame(pending);
      video.pause();
      video.removeAttribute("src");
      letGo();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the panel:`, error);
    }
  };

  return { element: root, height: PANEL_HEIGHT, maxHeight: Number.MAX_SAFE_INTEGER, minWidth: PANEL_MIN_WIDTH, dispose, refresh };
}

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_video_scopes_wrapped) return;
    proto.__was_video_scopes_wrapped = true;
    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createScopesPanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          panel.dispose();
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the panel:`, error);
      }
      return result;
    };
  },
});
