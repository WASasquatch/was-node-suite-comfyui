/**
 * The before and after audio panel on the nodes that change one clip, and the result panel on
 * the nodes that join clips.
 *
 * Each node adds what it changed: fades, a seam, loop seams, silence taken out or joins.
 */

import { app } from "../../scripts/app.js";
import { createGraphSection } from "./interface/audio_graph.js";
import { chainWidgetCallback } from "./interface/widget.js";
import {
  curveGain,
  drawEnvelope,
  drawMarkers,
  mountAudioPanel,
  readNumber,
  drawLabel,
  roundTime,
  shadeSpan,
  findWidget,
} from "./interface/audio_view.js";

const EXT_NAME = "WASNodeSuite.AudioBeforeAfter";
const SETTING_ID = "WAS.Audio.ShowPanels";

// Widgets whose hand edits move something the panel draws.
const WATCHED = ["fade_in", "fade_out", "curve_in", "curve_out", "duration", "curve"];

// Pixels a handle can be grabbed from.
const GRAB = 7;

/** The gain a pair of end fades applies at a time. */
function fadeGain(seconds, total, fadeIn, fadeOut, curveIn, curveOut) {
  let gain = 1;
  if (fadeIn > 0 && seconds < fadeIn) gain *= curveGain(curveIn, seconds / fadeIn);
  if (fadeOut > 0 && seconds > total - fadeOut) gain *= curveGain(curveOut, (total - seconds) / fadeOut);
  return gain;
}

/** The fade widgets as they stand, or as a drag holds them. */
function fadeSettings(panel, held) {
  const node = panel.node;
  return {
    fadeIn: held?.fade_in ?? readNumber(node, "fade_in"),
    fadeOut: held?.fade_out ?? readNumber(node, "fade_out"),
    curveIn: String(findWidget(node, "curve_in")?.value ?? "s-curve"),
    curveOut: String(findWidget(node, "curve_out")?.value ?? "s-curve"),
  };
}

/** Fade handles at the top corners of the faded spans, dragged to set each length. */
function fadeTool(panel) {
  let dragging = null;
  let held = null;
  const handles = () => {
    const total = panel.clip()?.duration ?? 0;
    const { fadeIn, fadeOut } = fadeSettings(panel, held);
    const top = panel.lanes()[0]?.top ?? 0;
    return [
      { name: "fade_in", x: panel.xAt(Math.min(total, fadeIn)), y: top + 5 },
      { name: "fade_out", x: panel.xAt(Math.max(0, total - fadeOut)), y: top + 5 },
    ];
  };
  const near = (at) => handles().find((handle) => Math.abs(handle.x - at.x) <= GRAB && Math.abs(handle.y - at.y) <= GRAB * 2);
  return {
    held: () => held,
    handles,
    cursor: (at) => (near(at) ? "ew-resize" : ""),
    pointerDown(event, at) {
      const handle = near(at);
      if (!handle) return false;
      dragging = handle.name;
      held = { fade_in: readNumber(panel.node, "fade_in"), fade_out: readNumber(panel.node, "fade_out") };
      return true;
    },
    pointerMove(event, at) {
      if (!dragging) return;
      const total = panel.clip()?.duration ?? 0;
      held[dragging] = roundTime(dragging === "fade_in" ? Math.min(total, at.time) : Math.max(0, total - at.time));
      panel.repaint();
    },
    pointerUp() {
      if (dragging && held) panel.write(dragging, held[dragging]);
      dragging = null;
      held = null;
      panel.repaint();
    },
  };
}

function fadeLayer(ctx, frame, panel) {
  const total = panel.clip()?.duration ?? 0;
  const settings = fadeSettings(panel, panel.tool?.held?.());
  ctx.setLineDash(panel.side() === "in" ? [4, 3] : []);
  drawEnvelope(ctx, frame, (time) => fadeGain(time, total, settings.fadeIn, settings.fadeOut,
    settings.curveIn, settings.curveOut), frame.theme.warning);
  ctx.setLineDash([]);
  for (const handle of panel.tool?.handles?.() ?? []) {
    ctx.fillStyle = frame.theme.warning;
    ctx.fillRect(handle.x - 4, handle.y - 4, 8, 8);
  }
}

/** The seam's left edge, dragged to set how long the two clips overlap. */
function seamTool(panel) {
  let held = null;
  const seam = () => {
    const facts = panel.report()?.transition;
    if (!Array.isArray(facts?.seam)) return null;
    const end = facts.seam[1];
    const start = held === null ? facts.seam[0] : end - held;
    return { start, end, longest: Math.min(facts.a_duration ?? end, facts.b_duration ?? end) };
  };
  const near = (at) => {
    const span = seam();
    return span && Math.abs(panel.xAt(span.start) - at.x) <= GRAB;
  };
  return {
    seam,
    cursor: (at) => (near(at) ? "ew-resize" : ""),
    pointerDown(event, at) {
      if (!near(at)) return false;
      const span = seam();
      held = span.end - span.start;
      return true;
    },
    pointerMove(event, at) {
      const span = seam();
      if (!span || held === null) return;
      held = roundTime(Math.max(0, Math.min(span.longest, span.end - at.time)));
      panel.repaint();
    },
    pointerUp() {
      if (held !== null) panel.write("duration", held);
      held = null;
      panel.repaint();
    },
  };
}

function seamLayer(ctx, frame, panel) {
  const span = panel.tool?.seam?.();
  if (!span) return;
  const curve = String(panel.report()?.transition?.curve ?? "equal power");
  shadeSpan(ctx, frame, span.start, span.end, frame.theme.accent, 0.14);
  const length = Math.max(1e-6, span.end - span.start);
  drawEnvelope(ctx, frame, (time) => (time < span.start ? 1 : curveGain(curve, (span.end - time) / length)),
    frame.theme.warning, [span.start - length * 0.15, span.end]);
  drawEnvelope(ctx, frame, (time) => (time > span.end ? 1 : curveGain(curve, (time - span.start) / length)),
    frame.theme.success, [span.start, span.end + length * 0.15]);
  ctx.fillStyle = frame.theme.fg;
  ctx.fillRect(Math.round(frame.xAt(span.start)) - 1, frame.top, 3, frame.height - frame.top);
}

function loopLayer(ctx, frame, panel) {
  if (panel.side() !== "out") return;
  drawMarkers(ctx, frame, panel.report()?.loop?.seams ?? [], frame.theme.warning, { dashed: true });
}

function joinLayer(ctx, frame, panel) {
  drawMarkers(ctx, frame, panel.report()?.join?.joins ?? [], frame.theme.warning, { dashed: true });
}

function silenceLayer(ctx, frame, panel) {
  if (panel.side() !== "in") return;
  for (const [from, to] of panel.report()?.silence?.removed ?? []) {
    shadeSpan(ctx, frame, from, to, frame.theme.error, 0.22);
  }
}

function padLayer(ctx, frame, panel) {
  if (panel.side() !== "out") return;
  const start = Number(panel.report()?.pad?.start) || 0;
  const length = panel.report()?.in?.duration ?? 0;
  shadeSpan(ctx, frame, 0, start, frame.theme.fgMuted, 0.18);
  shadeSpan(ctx, frame, start + length, panel.clip()?.duration ?? 0, frame.theme.fgMuted, 0.18);
}

function normalizeFigures(report) {
  const facts = report?.normalize;
  if (!facts) return undefined;
  const unit = facts.mode === "loudness" ? "LUFS" : facts.mode === "true peak" ? "dBTP" : "dBFS";
  const gain = Number(facts.gain_db?.[0] ?? 0);
  const before = facts.measured?.[0];
  const from = typeof before === "number" ? before.toFixed(1) : "silent";
  const clips = (facts.gain_db?.length ?? 0) > 1 ? ` · ${facts.gain_db.length} clips` : "";
  return `${gain >= 0 ? "+" : ""}${gain.toFixed(2)} dB · ${from} to ${Number(facts.target).toFixed(1)} ${unit}${clips}`;
}

function stretchMap(seconds, from, to, report) {
  const speed = Number(report?.stretch?.speed) || 1;
  return from === "in" && to === "out" ? seconds / speed : from === "out" && to === "in" ? seconds * speed : seconds;
}

function reverseMap(seconds, from, to, report) {
  const total = Number(report?.[to]?.duration) || 0;
  return from === to ? seconds : Math.max(0, total - seconds);
}

function padMap(seconds, from, to, report) {
  const start = Number(report?.pad?.start) || 0;
  return from === "in" && to === "out" ? seconds + start : from === "out" && to === "in" ? seconds - start : seconds;
}

/** The saturation curve a Saturator applied, input across and output up, from the node's own points. */
function saturationSection(node) {
  const section = createGraphSection({
    height: 120,
    hint: "The curve the samples went through: input across, output up, at the drive and bias set",
    draw(ctx, size, theme) {
      const facts = node.__wasAudioPanel?.report()?.saturator;
      const side = Math.min(size.width, size.height) - 10;
      const left = (size.width - side) / 2;
      const top = 5;
      ctx.strokeStyle = theme.border;
      ctx.lineWidth = 1;
      ctx.strokeRect(left + 0.5, top + 0.5, side - 1, side - 1);
      ctx.beginPath();
      ctx.moveTo(left, top + side / 2);
      ctx.lineTo(left + side, top + side / 2);
      ctx.moveTo(left + side / 2, top);
      ctx.lineTo(left + side / 2, top + side);
      ctx.globalAlpha = 0.5;
      ctx.stroke();
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(left, top + side);
      ctx.lineTo(left + side, top);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      const frame = { width: size.width, height: size.height, top: 0, theme };
      if (!facts?.curve?.length) {
        drawLabel(ctx, frame, "run to see the curve", left + 6, top + side / 2 - 12, { muted: true });
        return;
      }
      const peak = Math.max(1, ...facts.curve.map(Math.abs));
      ctx.strokeStyle = theme.accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      facts.curve.forEach((value, index) => {
        const px = left + (index / (facts.curve.length - 1)) * side;
        const py = top + side / 2 - (value / peak) * (side / 2);
        if (index === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      drawLabel(ctx, frame, `${facts.mode}, +${Number(facts.drive_db).toFixed(1)} dB`, left + side - 4, top + 10, { align: "right" });
    },
  });
  return { element: section.element, update: () => section.repaint(), dispose: () => section.dispose() };
}

function crusherFigures(report) {
  const facts = report?.crusher;
  if (!facts) return undefined;
  const rate = facts.sample_rate >= 1000 ? `${(facts.sample_rate / 1000).toFixed(2)} kHz` : `${facts.sample_rate.toFixed(0)} Hz`;
  return `${Number(facts.bits).toFixed(1)} bits at ${rate}${facts.voicing !== "custom" ? ` (${facts.voicing})` : ""}`;
}

const figuresFrom = (key, words) => (report) => (report?.[key] ? words(report[key]) : undefined);

/** One cycle of the LFO a modulation node ran, with its rate. */
/** The LFO across the clip's own timeline, in a band under the waveform, as the node ran it. */
function lfoLayer(key) {
  return (ctx, frame, panel) => {
    const facts = panel.report()?.[key];
    const points = facts?.cycle;
    const rate = Number(facts?.rate_hz);
    if (!Array.isArray(points) || points.length < 2 || !(rate > 0)) return;
    const band = Math.max(20, (frame.height - frame.top) * 0.26);
    const top = frame.height - band;
    const middle = top + band / 2;
    const reach = band / 2 - 3;
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = frame.theme.bgDark ?? frame.theme.bg;
    ctx.fillRect(0, top, frame.width, band);
    ctx.globalAlpha = 1;
    const count = points.length;
    const steps = Array.isArray(facts.steps) ? facts.steps : null;
    const random = String(facts.shape ?? "").startsWith("random");
    const valueAt = (seconds) => {
      const cycles = Math.max(0, seconds) * rate;
      const whole = Math.floor(cycles);
      const fraction = cycles - whole;
      if (steps) {
        const here = steps[Math.min(steps.length - 1, whole)];
        if (facts.shape === "random steps") return here;
        const there = steps[Math.min(steps.length - 1, whole + 1)];
        return here + (there - here) * (0.5 - 0.5 * Math.cos(Math.PI * fraction));
      }
      const at = fraction * count;
      const index = Math.floor(at);
      const share = at - index;
      return points[index % count] * (1 - share) + points[(index + 1) % count] * share;
    };
    const perPixel = (frame.view.t1 - frame.view.t0) / Math.max(1, frame.width);
    ctx.strokeStyle = frame.theme.warning;
    if (rate * perPixel > 0.25 || (random && !steps)) {
      ctx.globalAlpha = 0.45;
      ctx.fillStyle = frame.theme.warning;
      ctx.fillRect(0, middle - reach, frame.width, reach * 2);
      ctx.globalAlpha = 1;
    } else {
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let px = 0; px <= frame.width; px += 1) {
        const py = middle - valueAt(frame.timeAt(px)) * reach;
        if (px === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    const speed = rate >= 1 ? rate.toFixed(2) : rate.toFixed(3);
    const sync = facts.synced ? `, ${facts.division} at ${facts.bpm} bpm` : "";
    drawLabel(ctx, frame, `LFO ${facts.shape ?? ""} ${speed} Hz${sync}`, 4, top + 9, { muted: true, swatch: frame.theme.warning });
  };
}

/** A reverb's energy decay, drawn against the -60 dB its RT60 is measured to. */
function decaySection(key) {
  return (node) => {
    const section = createGraphSection({
      height: 90,
      hint: "Energy decay of the reverb's own impulse response, in the 1 kHz octave",
      draw(ctx, size, theme) {
        const facts = node.__wasAudioPanel?.report()?.[key];
        const frame = { width: size.width, height: size.height, top: 0, theme };
        const curve = facts?.decay;
        const floor = -70;
        const yOf = (db) => 4 + (Math.max(floor, Math.min(0, db)) / floor) * (size.height - 16);
        ctx.strokeStyle = theme.border;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(0, Math.round(yOf(-60)) + 0.5);
        ctx.lineTo(size.width, Math.round(yOf(-60)) + 0.5);
        ctx.stroke();
        ctx.setLineDash([]);
        if (!curve?.values?.length) {
          drawLabel(ctx, frame, "run to see the decay", 6, size.height / 2, { muted: true });
          return;
        }
        const span = curve.values.length * curve.step;
        ctx.strokeStyle = theme.accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        curve.values.forEach((db, index) => {
          const px = ((index * curve.step) / span) * size.width;
          if (index === 0) ctx.moveTo(px, yOf(db));
          else ctx.lineTo(px, yOf(db));
        });
        ctx.stroke();
        drawLabel(ctx, frame, "-60 dB", 4, yOf(-60) - 9, { muted: true });
        const measured = typeof facts.rt60 === "number" ? `RT60 ${facts.rt60.toFixed(2)} s` : "RT60 not measurable";
        const target = typeof facts.target === "number" ? ` (set ${facts.target.toFixed(2)} s)` : "";
        drawLabel(ctx, frame, `${measured}${target}`, size.width - 4, 10, { align: "right" });
        drawLabel(ctx, frame, `${span.toFixed(2)} s`, size.width - 4, size.height - 9, { align: "right", muted: true });
      },
    });
    return { element: section.element, update: () => section.repaint(), dispose: () => section.dispose() };
  };
}

/** A delay's repeats as bars at their times, as tall as their level. */
function echoSection(node) {
  const section = createGraphSection({
    height: 70,
    hint: "The repeats: when each lands and how loud it is",
    draw(ctx, size, theme) {
      const facts = node.__wasAudioPanel?.report()?.delay;
      const frame = { width: size.width, height: size.height, top: 0, theme };
      if (!facts?.repeats?.length) {
        drawLabel(ctx, frame, "run to see the repeats", 6, size.height / 2, { muted: true });
        return;
      }
      const last = facts.repeats[facts.repeats.length - 1] * 1.08;
      const base = size.height - 14;
      ctx.strokeStyle = theme.border;
      ctx.beginPath();
      ctx.moveTo(0, base + 0.5);
      ctx.lineTo(size.width, base + 0.5);
      ctx.stroke();
      ctx.fillStyle = theme.fg;
      ctx.fillRect(2, 6, 3, base - 6);
      facts.repeats.forEach((time, index) => {
        const level = Array.isArray(facts.levels) ? Number(facts.levels[index]) : -6 * (index + 1);
        const tall = Math.max(2, (1 + Math.max(-60, level) / 60) * (base - 6));
        const px = 2 + (time / last) * (size.width - 8);
        const side = facts.sides?.[index];
        ctx.fillStyle = side === "left" ? theme.success : side === "right" ? theme.warning : theme.accent;
        ctx.fillRect(px, base - tall, 3, tall);
      });
      drawLabel(ctx, frame, `${(facts.time * 1000).toFixed(1)} ms, ${facts.mode}`, size.width - 4, 10, { align: "right" });
      if (facts.sides?.length) {
        drawLabel(ctx, frame, "left", size.width - 4, 28, { align: "right", swatch: theme.success, muted: true });
        drawLabel(ctx, frame, "right", size.width - 52, 28, { align: "right", swatch: theme.warning, muted: true });
      }
    },
  });
  return { element: section.element, update: () => section.repaint(), dispose: () => section.dispose() };
}

const hz = (value) => (Number(value) >= 1000 ? `${(Number(value) / 1000).toFixed(1)} kHz` : `${Number(value).toFixed(0)} Hz`);

// Each node's additions to the shared panel.
const FAMILY = {
  WASAudioNormalize: { figures: normalizeFigures },
  WASAudioFade: { layers: [fadeLayer], tool: fadeTool },
  WASAudioTimeStretch: { mapTime: stretchMap },
  WASAudioLoop: { layers: [loopLayer] },
  WASAudioReverse: { mapTime: reverseMap },
  WASAudioPad: { layers: [padLayer], mapTime: padMap },
  WASAudioResample: {},
  WASAudioChannels: {},
  WASAudioRemoveSilence: { layers: [silenceLayer], initial: "out" },
  WASAudioJoin: { sides: [{ key: "out", label: "Joined" }], layers: [joinLayer] },
  WASAudioTransition: { sides: [{ key: "out", label: "Result" }], layers: [seamLayer], tool: seamTool },
  WASAudioSaturator: { initial: "out", height: 330, extra: saturationSection },
  WASAudioBitcrusher: { initial: "out", figures: crusherFigures },
  WASAudioLoFi: { initial: "out", figures: figuresFrom("lofi", (f) => `${f.medium}, age ${Number(f.age).toFixed(2)}`) },
  WASAudioSlew: {
    initial: "out",
    figures: figuresFrom("slew", (f) => `${f.mode}: rise ${Number(f.rise_ms).toFixed(3)} ms, fall ${Number(f.fall_ms).toFixed(3)} ms`),
  },
  WASAudioRingMod: {
    initial: "out",
    figures: figuresFrom("ring", (f) => `${f.mode} at ${Number(f.frequency).toFixed(1)} Hz${f.lfo_depth ? `, wandering ${Number(f.lfo_depth).toFixed(0)} Hz` : ""}`),
  },
  WASAudioChorus: {
    initial: "out",
    figures: figuresFrom("chorus", (f) => `${f.mode}, ${f.voices} voices at ${Number(f.rate_hz).toFixed(2)} Hz, `
      + `${Number(f.pitch_cents ?? 0).toFixed(1)} cents of wobble`),
  },
  WASAudioFlanger: {
    initial: "out",
    height: 240,
    layers: [lfoLayer("flanger")],
    figures: figuresFrom("flanger", (f) => `${f.mode}, ${Number(f.min_ms).toFixed(2)} to ${Number(f.max_ms).toFixed(2)} ms`
      + (f.notch_low_hz && f.notch_high_hz ? `, first notch ${hz(f.notch_low_hz)} to ${hz(f.notch_high_hz)}` : "")),
  },
  WASAudioPhaser: {
    initial: "out",
    height: 240,
    layers: [lfoLayer("phaser")],
    figures: figuresFrom("phaser", (f) => `${f.stages} stages sweeping ${hz(f.low_hz)} to ${hz(f.high_hz)}, ${f.notches} notches`),
  },
  WASAudioTremolo: {
    initial: "out",
    height: 240,
    layers: [lfoLayer("tremolo")],
    figures: figuresFrom("tremolo", (f) => `${f.mode}, depth ${Number(f.depth).toFixed(2)}`
      + (typeof f.floor_db === "number" ? `, dips to ${f.floor_db.toFixed(1)} dB` : "")),
  },
  WASAudioReverb: {
    initial: "out",
    height: 300,
    extra: decaySection("reverb"),
    figures: figuresFrom("reverb", (f) => `${f.room ?? "reverb"}, RT60 ${typeof f.rt60 === "number" ? f.rt60.toFixed(2) : "?"} s`),
  },
  WASAudioConvolutionReverb: {
    initial: "out",
    height: 300,
    extra: decaySection("convolution"),
    figures: figuresFrom("convolution", (f) => `impulse ${Number(f.impulse_seconds).toFixed(2)} s, RT60 `
      + `${typeof f.rt60 === "number" ? f.rt60.toFixed(2) : "?"} s`),
  },
  WASAudioDelay: {
    initial: "out",
    height: 280,
    extra: echoSection,
    figures: figuresFrom("delay", (f) => `${(f.time * 1000).toFixed(1)} ms, feedback ${Number(f.feedback).toFixed(2)}`),
  },
  WASAudioPitchShift: {
    initial: "out",
    figures: figuresFrom("pitch", (f) => `${f.semitones >= 0 ? "+" : ""}${Number(f.semitones).toFixed(2)} semitones, formants ${f.keep_formants ? "kept" : "moved"}`),
  },
};

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Audio", "Show audio panels"],
      name: "Draw the audio panels",
      tooltip:
        "Draw each audio node's waveform, its playback controls and its level meter, with the "
        + "clip before and after the node on the nodes that change one. Off, the nodes run the "
        + "same and draw nothing. This applies to nodes added after the setting changes, so a "
        + "reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const family = FAMILY[nodeData?.name];
    if (!family) return;
    const proto = nodeType.prototype;
    if (proto.__was_audio_before_after_wrapped) return;
    proto.__was_audio_before_after_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const node = this;
        const options = { ...family };
        if (family.extra) options.extra = () => family.extra(node);
        const built = mountAudioPanel(this, options);
        if (built) {
          node.__wasAudioPanel = built.panel;
          for (const name of WATCHED) chainWidgetCallback(this, name, () => built.panel.repaint(), EXT_NAME);
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the audio panel:`, error);
      }
      return result;
    };
  },
});
