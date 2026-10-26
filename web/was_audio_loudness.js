/**
 * The loudness meter drawn on Audio Loudness: the figures a delivery is checked against, and the
 * momentary and short-term loudness drawn over the waveform against the target.
 */

import { app } from "../../scripts/app.js";
import { drawLabel, findWidget, mountAudioPanel, readNumber } from "./interface/audio_view.js";
import { createFigureTile } from "./interface/report_panel.js";
import { themeVar } from "./interface/theme.js";
import { chainWidgetCallback } from "./interface/widget.js";
import { el } from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.AudioLoudness";
const NODE_ID = "WASAudioLoudness";

// The loudness the graph spans, in LUFS, bottom to top.
const GRAPH_FLOOR = -60;
const GRAPH_TOP = 0;

// What a figure reads at or below the measured floor.
const SILENT = -119.9;

const lufs = (value, unit) => (typeof value === "number" && value > SILENT ? `${value.toFixed(1)} ${unit}` : "silent");

function figureRow() {
  const element = el("div", "flex:0 0 auto;display:flex;flex-wrap:wrap;gap:4px 14px;padding:2px 2px 0");
  const update = (report) => {
    const reading = report?.loudness;
    element.replaceChildren();
    element.style.display = reading ? "flex" : "none";
    if (!reading) return;
    const over = typeof reading.target === "number" && reading.integrated > SILENT
      ? reading.integrated - reading.target : null;
    const peak = createFigureTile("true peak", lufs(reading.true_peak, "dBTP"),
      "Highest point between samples; above -1 dBTP risks clipping once encoded");
    if (reading.true_peak > -1) peak.firstElementChild?.style.setProperty("color", themeVar("error"));
    element.append(
      createFigureTile("integrated", lufs(reading.integrated, "LUFS"),
        over === null ? "" : `${over >= 0 ? "+" : ""}${over.toFixed(1)} LU against the ${reading.target.toFixed(1)} LUFS target`),
      createFigureTile("range", `${Number(reading.range).toFixed(1)} LU`, "Loudness range: the spread of the loud and quiet passages"),
      peak,
      createFigureTile("short-term max", lufs(reading.short_term_max, "LUFS"), "Loudest 3 seconds"),
      createFigureTile("momentary max", lufs(reading.momentary_max, "LUFS"), "Loudest 400 milliseconds"),
    );
  };
  return { element, update };
}

function curveLayer(ctx, frame, panel) {
  const reading = panel.report()?.loudness;
  if (!reading?.series) return;
  const top = frame.top + 2;
  const height = frame.height - top - 2;
  const yAt = (value) => top + (1 - (Math.max(GRAPH_FLOOR, Math.min(GRAPH_TOP, value)) - GRAPH_FLOOR)
    / (GRAPH_TOP - GRAPH_FLOOR)) * height;
  const draw = (curve, colour, width) => {
    if (!curve?.values?.length) return;
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    let started = false;
    curve.values.forEach((value, index) => {
      const x = frame.xAt(curve.offset + index * curve.step);
      if (x < -2 || x > frame.width + 2) return;
      const y = yAt(value);
      if (!started) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      started = true;
    });
    ctx.stroke();
  };
  ctx.fillStyle = frame.theme.bgDark ?? frame.theme.bg;
  ctx.globalAlpha = 0.45;
  ctx.fillRect(0, top, frame.width, height);
  ctx.globalAlpha = 1;
  draw(reading.series.momentary, frame.theme.fgMuted, 1);
  draw(reading.series.short_term, frame.theme.warning, 2);
  const target = readNumber(panel.node, "target", reading.target ?? -14);
  const y = Math.round(yAt(target)) + 0.5;
  ctx.strokeStyle = frame.theme.success;
  ctx.setLineDash([4, 3]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(frame.width, y);
  ctx.stroke();
  ctx.setLineDash([]);
  for (const level of [-12, -24, -36, -48]) {
    if (Math.abs(level - target) < 4) continue;
    drawLabel(ctx, frame, String(level), frame.width - 3, yAt(level), { align: "right", muted: true });
  }
  // The target sits above its line, or below it where the line runs along the top.
  const above = y - 9 > top + 7;
  drawLabel(ctx, frame, `target ${target.toFixed(1)} LUFS`, 4, above ? y - 10 : y + 10, { swatch: frame.theme.success });
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_audio_loudness_wrapped) return;
    proto.__was_audio_loudness_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const built = mountAudioPanel(this, {
          sides: [{ key: "in", label: "Clip" }],
          initial: "in",
          height: 240,
          layers: [curveLayer],
          extra: () => figureRow(),
          figures: (report) => (report?.loudness
            ? "thick line short-term, thin line momentary, dashed line target" : undefined),
        });
        const target = findWidget(this, "target");
        if (built && target) chainWidgetCallback(this, target.name, () => built.panel.repaint(), EXT_NAME);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the loudness meter:`, error);
      }
      return result;
    };
  },
});
