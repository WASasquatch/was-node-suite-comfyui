/**
 * The Audio Analog Filter panel: the voicing's response with cutoff and resonance on one dot to
 * drag, and the cutoff the envelope and LFO moved, traced over the waveform.
 *
 * The curve is the small-signal response, saturation left out.
 */

import { app } from "../../scripts/app.js";
import { LOG_LOW, createGraphSection, logHz, logX } from "./interface/audio_graph.js";
import { drawLabel, findWidget, mountAudioPanel, readNumber } from "./interface/audio_view.js";
import { chainWidgetCallback } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.AudioAnalogFilter";
const NODE_ID = "WASAudioAnalogFilter";

const MODES = {
  ladder: ["low pass", "high pass", "band pass"],
  diode: ["low pass"],
  "ms-20": ["low pass", "high pass"],
  sem: ["low pass", "high pass", "band pass", "notch"],
};
const TOP_DB = 24;
const BOTTOM_DB = -48;
const GRAB = 10;

/** The small-signal response of a voicing at one frequency, the node's own formula. */
export function responseDb(model, mode, cutoff, resonance, hertz) {
  const w = hertz / Math.max(1, cutoff);
  const reso = Math.max(0, Math.min(1, resonance));
  const pick = (MODES[model] ?? ["low pass"]).includes(mode) ? mode : "low pass";
  // Complex arithmetic on s = jw, as [real, imaginary].
  const s = [0, w];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  const scale = (a, k) => [a[0] * k, a[1] * k];
  const power = (a, n) => Array.from({ length: n }).reduce((acc) => mul(acc, a), [1, 0]);
  const size = (a) => Math.hypot(a[0], a[1]);
  let num;
  let den;
  if (model === "sem") {
    const k = 2 - 1.96 * reso;
    den = add(add(mul(s, s), scale(s, k)), [1, 0]);
    num = { "low pass": [1, 0], "high pass": mul(s, s), "band pass": s, notch: add(mul(s, s), [1, 0]) }[pick];
  } else if (model === "ms-20") {
    const k = 2.2 * reso * 0.9;
    den = add(add(mul(s, s), scale(s, 2 - k)), [1, 0]);
    num = pick === "high pass" ? mul(s, s) : [1, 0];
  } else if (model === "diode") {
    const k = 3.6 * reso * 0.55;
    den = add(power(add([1, 0], s), 3), [k, 0]);
    num = [1 + k, 0];
  } else {
    const k = 4 * reso;
    den = add(power(add([1, 0], s), 4), [k, 0]);
    num = { "low pass": [1, 0], "high pass": power(s, 4), "band pass": scale(mul(s, s), 4) }[pick];
  }
  return 20 * Math.log10(Math.max(1e-9, size(num) / Math.max(1e-12, size(den))));
}

function responseSection(node) {
  let held = null;
  let dragging = false;
  const box = { width: 1, height: 1 };
  const yOf = (db) => ((TOP_DB - Math.max(BOTTOM_DB, Math.min(TOP_DB, db))) / (TOP_DB - BOTTOM_DB)) * (box.height - 16) + 4;
  const settings = () => ({
    model: String(findWidget(node, "model")?.value ?? "ladder"),
    mode: String(findWidget(node, "mode")?.value ?? "low pass"),
    cutoff: held?.cutoff ?? readNumber(node, "cutoff", 1200),
    resonance: held?.resonance ?? readNumber(node, "resonance", 0.3),
  });
  const dot = () => {
    const s = settings();
    return { px: logX(s.cutoff, box.width), py: 6 + (1 - s.resonance) * (box.height - 22) };
  };
  return createGraphSection({
    height: 120,
    hint: "Drag the dot: left and right for cutoff, up and down for resonance",
    draw(ctx, size, theme) {
      box.width = size.width;
      box.height = size.height;
      const s = settings();
      ctx.strokeStyle = theme.border;
      ctx.lineWidth = 1;
      ctx.font = "9px system-ui,sans-serif";
      for (const hz of [50, 100, 200, 500, 1000, 2000, 5000, 10000]) {
        const px = Math.round(logX(hz, size.width)) + 0.5;
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, size.height);
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = theme.fgMuted;
        ctx.fillText(hz >= 1000 ? `${hz / 1000}k` : String(hz), px + 2, size.height - 3);
      }
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.moveTo(0, Math.round(yOf(0)) + 0.5);
      ctx.lineTo(size.width, Math.round(yOf(0)) + 0.5);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = theme.accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let px = 0; px <= size.width; px += 1.5) {
        const py = yOf(responseDb(s.model, s.mode, s.cutoff, s.resonance, logHz(px, size.width)));
        if (px === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      const spot = dot();
      ctx.fillStyle = theme.warning;
      ctx.beginPath();
      ctx.arc(spot.px, spot.py, 5.5, 0, Math.PI * 2);
      ctx.fill();
      const frame = { width: size.width, height: size.height, top: 0, theme };
      const hz = s.cutoff >= 1000 ? `${(s.cutoff / 1000).toFixed(2)} kHz` : `${s.cutoff.toFixed(0)} Hz`;
      drawLabel(ctx, frame, `${s.model} ${s.mode} · ${hz} · resonance ${s.resonance.toFixed(2)}`, size.width - 6, 12, { align: "right" });
    },
    pointer: {
      cursor: (at) => {
        const spot = dot();
        return Math.hypot(spot.px - at.x, spot.py - at.y) <= GRAB ? "grab" : "crosshair";
      },
      down(at) {
        dragging = true;
        held = {};
        this.move(at);
        return true;
      },
      move(at) {
        if (!dragging) return;
        held.cutoff = Math.round(Math.max(LOG_LOW, Math.min(20000, logHz(at.x, box.width))));
        held.resonance = Math.round(Math.max(0, Math.min(1, 1 - (at.y - 6) / Math.max(1, box.height - 22))) * 100) / 100;
      },
      up() {
        const panel = node.__wasAudioPanel;
        if (held && panel) {
          if ("cutoff" in held) panel.write("cutoff", held.cutoff);
          if ("resonance" in held) panel.write("resonance", held.resonance);
        }
        dragging = false;
        held = null;
      },
    },
  });
}

function cutoffLayer(ctx, frame, panel) {
  const curve = panel.report()?.filter?.cutoff_curve;
  if (!curve?.values?.length || curve.values.every((value) => value === curve.values[0])) return;
  const top = frame.top + 2;
  const height = frame.height - top - 4;
  const yOf = (hz) => top + (1 - Math.log(hz / LOG_LOW) / Math.log(20000 / LOG_LOW)) * height;
  ctx.strokeStyle = frame.theme.warning;
  ctx.lineWidth = 2;
  ctx.beginPath();
  curve.values.forEach((hz, index) => {
    const px = frame.xAt(curve.offset + (index + 0.5) * curve.step);
    if (index === 0) ctx.moveTo(px, yOf(hz));
    else ctx.lineTo(px, yOf(hz));
  });
  ctx.stroke();
  drawLabel(ctx, frame, "cutoff", 4, top + 9, { muted: true, swatch: frame.theme.warning });
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_audio_analog_filter_wrapped) return;
    proto.__was_audio_analog_filter_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const node = this;
        let section = null;
        const built = mountAudioPanel(this, {
          initial: "out",
          height: 330,
          layers: [cutoffLayer],
          extra: () => {
            section = responseSection(node);
            return { element: section.element, update: () => section.repaint(), dispose: () => section.dispose() };
          },
        });
        if (built) {
          node.__wasAudioPanel = built.panel;
          for (const name of ["model", "mode", "cutoff", "resonance"]) {
            chainWidgetCallback(node, name, () => section?.repaint(), EXT_NAME);
          }
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the filter panel:`, error);
      }
      return result;
    };
  },
});
