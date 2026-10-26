/**
 * The dynamics panels: the gain each node applied, traced over the waveform, and its transfer
 * curve, whose threshold, ratio, depth or ceiling drags into place.
 */

import { app } from "../../scripts/app.js";
import { createGraphSection } from "./interface/audio_graph.js";
import { drawLabel, findWidget, mountAudioPanel, readNumber } from "./interface/audio_view.js";
import { chainWidgetCallback } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.AudioDynamics";

// The level range the transfer curve spans, in dBFS.
const FLOOR_DB = -60;

// The gain range the trace over the waveform spans, in dB either way.
const TRACE_DB = 24;

// Pixels a handle can be grabbed from.
const GRAB = 9;

// Each node's curve, the widgets it reads, and the handles that drag.
const KINDS = {
  WASAudioCompressor: "compressor",
  WASAudioLimiter: "limiter",
  WASAudioGate: "gate",
  WASAudioTransientShaper: "transient",
  WASAudioDeEsser: "de-esser",
  WASAudioMultibandCompressor: "multiband",
};

// Widgets whose hand edits move the curve.
const WATCHED = ["threshold_db", "ratio", "knee_db", "depth_db", "hysteresis_db", "ceiling_db", "gain_db", "mode",
  "range_db", "low_threshold_db", "low_ratio", "mid_threshold_db", "mid_ratio", "high_threshold_db", "high_ratio"];

/** A compressor's gain at a level, the node's own soft knee formula. */
export function compressorGain(level, threshold, ratio, knee) {
  const over = level - threshold;
  const slope = 1 / Math.max(1, ratio) - 1;
  if (knee > 0 && Math.abs(over) <= knee / 2) return (slope * (over + knee / 2) ** 2) / (2 * knee);
  return over > 0 ? slope * over : 0;
}

/** A downward expander's gain at a level, the node's own formula. */
export function expanderGain(level, threshold, ratio, depth, knee) {
  const under = threshold - level;
  const slope = Math.max(1, ratio) - 1;
  let gain = under > 0 ? -slope * under : 0;
  if (knee > 0 && Math.abs(under) <= knee / 2) gain = (-slope * (under + knee / 2) ** 2) / (2 * knee);
  return Math.max(gain, -Math.abs(depth));
}

/** The settings a curve is drawn from: the node's widgets, or what a drag holds. */
function settingsOf(node, kind, held) {
  const read = (name, fallback) => (held && name in held ? held[name] : readNumber(node, name, fallback));
  const mode = String(findWidget(node, "mode")?.value ?? "gate");
  return {
    kind: kind === "gate" && mode === "expander" ? "expander" : kind,
    threshold: read("threshold_db", -20),
    ratio: read("ratio", kind === "de-esser" ? 8 : 4),
    knee: kind === "de-esser" ? 4 : kind === "gate" ? 6 : read("knee_db", 6),
    depth: read(kind === "de-esser" ? "range_db" : "depth_db", 40),
    hysteresis: read("hysteresis_db", 4),
    ceiling: read("ceiling_db", -1),
    drive: read("gain_db", 0),
    bands: [
      ["low", read("low_threshold_db", -20), read("low_ratio", 3)],
      ["mid", read("mid_threshold_db", -20), read("mid_ratio", 2)],
      ["high", read("high_threshold_db", -24), read("high_ratio", 2.5)],
    ],
  };
}

/** Output level at an input level for one set of settings. */
function outputAt(level, s, band = null) {
  if (band) return level + compressorGain(level, band[1], band[2], 6);
  if (s.kind === "limiter") return Math.min(level + s.drive, s.ceiling);
  if (s.kind === "expander") return level + expanderGain(level, s.threshold, s.ratio, s.depth, 6);
  if (s.kind === "gate") return level >= s.threshold ? level : level - Math.abs(s.depth);
  if (s.kind === "de-esser") return level + Math.max(compressorGain(level, s.threshold, 8, 4), -Math.abs(s.depth));
  return level + compressorGain(level, s.threshold, s.ratio, s.knee);
}

function transferSection(node, kind) {
  let held = null;
  let dragging = null;
  const box = { width: 1, height: 1 };
  const pad = { left: 26, right: 8, top: 8, bottom: 16 };
  const x = (db) => pad.left + ((db - FLOOR_DB) / -FLOOR_DB) * (box.width - pad.left - pad.right);
  const y = (db) => pad.top + (1 - (Math.max(FLOOR_DB, Math.min(6, db)) - FLOOR_DB) / (-FLOOR_DB + 6)) * (box.height - pad.top - pad.bottom);
  const dbAtX = (px) => FLOOR_DB + ((px - pad.left) / Math.max(1, box.width - pad.left - pad.right)) * -FLOOR_DB;
  const dbAtY = (py) => FLOOR_DB + (1 - (py - pad.top) / Math.max(1, box.height - pad.top - pad.bottom)) * (-FLOOR_DB + 6);

  const handles = () => {
    const s = settingsOf(node, kind, held);
    if (s.kind === "multiband") return [];
    if (s.kind === "limiter") return [{ name: "ceiling_db", px: x(0), py: y(s.ceiling), axis: "y" }];
    const list = [{ name: "threshold_db", px: x(s.threshold), py: y(outputAt(s.threshold, s)), axis: "x" }];
    if (s.kind === "compressor") list.push({ name: "ratio", px: x(0), py: y(outputAt(0, s)), axis: "y" });
    if (s.kind === "gate" || s.kind === "expander" || s.kind === "de-esser") {
      const at = s.kind === "de-esser" ? 0 : Math.max(FLOOR_DB + 6, s.threshold - 24);
      list.push({ name: s.kind === "de-esser" ? "range_db" : "depth_db", px: x(at), py: y(outputAt(at, s)), axis: "y" });
    }
    return list;
  };

  const section = createGraphSection({
    height: 120,
    hint: "Drag the dots: threshold left and right; ratio, depth or ceiling up and down",
    draw(ctx, size, theme) {
      box.width = size.width;
      box.height = size.height;
      const s = settingsOf(node, kind, held);
      ctx.strokeStyle = theme.border;
      ctx.lineWidth = 1;
      ctx.font = "9px system-ui,sans-serif";
      ctx.fillStyle = theme.fgMuted;
      for (let db = FLOOR_DB; db <= 0; db += 12) {
        ctx.beginPath();
        ctx.moveTo(x(db), y(FLOOR_DB));
        ctx.lineTo(x(db), y(6));
        ctx.moveTo(x(FLOOR_DB), y(db));
        ctx.lineTo(x(0), y(db));
        ctx.globalAlpha = 0.4;
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillText(String(db), 2, y(db) + 3);
      }
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x(FLOOR_DB), y(FLOOR_DB));
      ctx.lineTo(x(0), y(0));
      ctx.stroke();
      ctx.setLineDash([]);
      const curves = s.kind === "multiband"
        ? s.bands.map((band, index) => [band, [theme.success, theme.warning, theme.accent][index]])
        : [[null, theme.accent]];
      for (const [band, colour] of curves) {
        ctx.strokeStyle = colour;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let db = FLOOR_DB; db <= 0.001; db += 0.25) {
          const px = x(db);
          const py = y(outputAt(db, s, band));
          if (db === FLOOR_DB) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
      if (s.kind === "gate") {
        ctx.fillStyle = theme.warning;
        ctx.globalAlpha = 0.15;
        ctx.fillRect(x(s.threshold - s.hysteresis), pad.top, x(s.threshold) - x(s.threshold - s.hysteresis), box.height - pad.top - pad.bottom);
        ctx.globalAlpha = 1;
      }
      for (const handle of handles()) {
        ctx.fillStyle = theme.warning;
        ctx.beginPath();
        ctx.arc(handle.px, handle.py, 4.5, 0, Math.PI * 2);
        ctx.fill();
      }
      const frame = { width: size.width, height: size.height, top: 0, theme };
      const words = s.kind === "limiter" ? `ceiling ${s.ceiling.toFixed(1)} dB`
        : s.kind === "multiband" ? "low, mid and high bands"
          : s.kind === "compressor" ? `${s.threshold.toFixed(1)} dB, ${s.ratio.toFixed(1)}:1`
            : `${s.threshold.toFixed(1)} dB, depth ${Math.abs(s.depth).toFixed(0)} dB`;
      drawLabel(ctx, frame, words, size.width - 6, pad.top + 9, { align: "right" });
      drawLabel(ctx, frame, "in", size.width - 6, size.height - 9, { align: "right", muted: true });
    },
    pointer: {
      cursor: (at) => (handles().some((h) => Math.hypot(h.px - at.x, h.py - at.y) <= GRAB) ? "grab" : ""),
      down(at) {
        const hit = handles().find((h) => Math.hypot(h.px - at.x, h.py - at.y) <= GRAB);
        if (!hit) return false;
        dragging = hit;
        held = {};
        return true;
      },
      move(at) {
        if (!dragging) return;
        const s = settingsOf(node, kind, held);
        if (dragging.name === "threshold_db") held.threshold_db = Math.round(Math.max(FLOOR_DB, Math.min(0, dbAtX(at.x))) * 2) / 2;
        else if (dragging.name === "ratio") {
          const out = dbAtY(at.y);
          const ratio = out > s.threshold ? (0 - s.threshold) / (out - s.threshold) : 30;
          held.ratio = Math.round(Math.max(1, Math.min(30, ratio)) * 10) / 10;
        } else if (dragging.name === "ceiling_db") held.ceiling_db = Math.round(Math.max(-24, Math.min(0, dbAtY(at.y))) * 10) / 10;
        else {
          const level = s.kind === "de-esser" ? 0 : Math.max(FLOOR_DB + 6, s.threshold - 24);
          const depth = Math.max(0, level - dbAtY(at.y));
          held[dragging.name] = Math.round(Math.min(dragging.name === "range_db" ? 24 : 90, depth) * 2) / 2;
        }
      },
      up() {
        if (dragging && held) {
          for (const [name, value] of Object.entries(held)) writeValue(node, name, value);
        }
        dragging = null;
        held = null;
      },
    },
  });
  return section;
}

/** Write a widget as one undoable edit. */
function writeValue(node, name, value) {
  const panel = node.__wasAudioPanel;
  if (panel) panel.write(name, value);
}

function traceLayer(ctx, frame, panel) {
  const curve = panel.report()?.dynamics?.gr;
  if (!curve?.values?.length) return;
  const top = frame.top + 1;
  const band = Math.max(18, (frame.height - top) * 0.32);
  const middle = top + band / 2;
  const positive = curve.values.some((value) => value > 0.05);
  const base = positive ? middle : top;
  const scale = positive ? band / 2 : band;
  ctx.globalAlpha = 0.85;
  ctx.fillStyle = frame.theme.bgDark ?? frame.theme.bg;
  ctx.fillRect(0, top, frame.width, band);
  ctx.globalAlpha = 1;
  ctx.beginPath();
  ctx.moveTo(0, base);
  curve.values.forEach((value, index) => {
    const px = frame.xAt(curve.offset + (index + 0.5) * curve.step);
    const share = Math.max(-1, Math.min(1, value / TRACE_DB));
    ctx.lineTo(px, base - share * scale);
  });
  ctx.lineTo(frame.width, base);
  ctx.closePath();
  ctx.fillStyle = frame.theme.warning;
  ctx.globalAlpha = 0.45;
  ctx.fill();
  ctx.globalAlpha = 1;
  drawLabel(ctx, frame, positive ? "gain change" : "gain reduction", 4, top + band - 9, { muted: true, swatch: frame.theme.warning });
}

function dynamicsFigures(report) {
  const facts = report?.dynamics;
  if (!facts) return undefined;
  if (facts.kind === "transient") return `largest change ${Math.max(...facts.gr.values.map(Math.abs), 0).toFixed(1)} dB`;
  const open = typeof facts.open_share === "number" ? ` · open ${(facts.open_share * 100).toFixed(0)}% of the time` : "";
  return `max reduction ${facts.max_gr.toFixed(1)} dB · average ${facts.avg_gr.toFixed(1)} dB${open}`;
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const kind = KINDS[nodeData?.name];
    if (!kind) return;
    const proto = nodeType.prototype;
    if (proto.__was_audio_dynamics_wrapped) return;
    proto.__was_audio_dynamics_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const node = this;
        let section = null;
        const built = mountAudioPanel(this, {
          initial: "out",
          height: kind === "transient" ? 200 : 330,
          layers: [traceLayer],
          figures: dynamicsFigures,
          extra: kind === "transient" ? undefined : () => {
            section = transferSection(node, kind);
            return { element: section.element, update: () => section.repaint(), dispose: () => section.dispose() };
          },
        });
        if (built) {
          node.__wasAudioPanel = built.panel;
          for (const name of WATCHED) {
            const widget = findWidget(node, name);
            if (widget) chainWidgetCallback(node, widget.name, () => section?.repaint(), EXT_NAME);
          }
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the dynamics panel:`, error);
      }
      return result;
    };
  },
});
