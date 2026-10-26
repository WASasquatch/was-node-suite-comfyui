/**
 * The equaliser panel on Audio Equalizer: the bands' combined response over the clip's spectrum
 * before and after, each band a dot to drag.
 *
 * The curve is computed from the same cookbook and Butterworth formulas the node runs.
 */

import { app } from "../../scripts/app.js";
import { LOG_LOW, createGraphSection, logHz, logX } from "./interface/audio_graph.js";
import { drawLabel, findWidget, mountAudioPanel } from "./interface/audio_view.js";
import { themeVar } from "./interface/theme.js";
import { chainWidgetCallback } from "./interface/widget.js";
import { button, el } from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.AudioEqualizer";
const NODE_ID = "WASAudioEqualizer";

const KINDS = ["bell", "low shelf", "high shelf", "low cut", "high cut", "notch", "band pass"];
const CUTS = ["low cut", "high cut"];
const GAINED = ["bell", "low shelf", "high shelf"];
const MAX_BANDS = 16;
const RANGE_DB = 24;
const GRAB = 9;

/** Bands read from the widget, or an error naming the entry that cannot be read. */
export function parseBands(text) {
  const bands = [];
  for (const raw of String(text ?? "").replace(/;/g, "\n").split("\n")) {
    let entry = raw.trim().toLowerCase();
    if (!entry || entry.startsWith("#")) continue;
    let on = true;
    if (entry.startsWith("off ")) {
      on = false;
      entry = entry.slice(4).trim();
    }
    const kind = [...KINDS].sort((a, b) => b.length - a.length).find((name) => entry.startsWith(name));
    if (!kind) return { error: `"${raw.trim()}" names no band kind` };
    const fields = entry.slice(kind.length).trim().split(/\s+/).filter(Boolean).map(Number);
    if (fields.some((value) => !Number.isFinite(value)) || !fields.length) return { error: `"${raw.trim()}" has a number that cannot be read` };
    if (CUTS.includes(kind)) bands.push({ kind, freq: fields[0], slope: Math.round((fields[1] ?? 12) / 6) * 6, gain: 0, q: 0.707, on });
    else bands.push({ kind, freq: fields[0], gain: fields[1] ?? 0, q: fields[2] ?? 0.707, slope: 12, on });
    if (bands.length > MAX_BANDS) return { error: `more than ${MAX_BANDS} bands; the equalizer holds at most ${MAX_BANDS}` };
  }
  return { bands };
}

/** Bands as the widget's text. */
export function writeBands(bands) {
  const freq = (value) => (value >= 1000 ? String(Math.round(value)) : String(Math.round(value * 10) / 10));
  return bands.map((band) => {
    const lead = band.on ? "" : "off ";
    if (CUTS.includes(band.kind)) return `${lead}${band.kind} ${freq(band.freq)} ${band.slope}`;
    return `${lead}${band.kind} ${freq(band.freq)} ${(Math.round(band.gain * 10) / 10)} ${(Math.round(band.q * 100) / 100)}`;
  }).join("; ");
}

/** A cookbook biquad's coefficients, the node's own formulas. */
function biquad(kind, freq, q, gainDb, rate) {
  const f = Math.min(Math.max(1, freq), 0.49 * rate);
  const Q = Math.max(0.025, q);
  const A = 10 ** (gainDb / 40);
  const w0 = (2 * Math.PI * f) / rate;
  const cos = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * Q);
  let b0, b1, b2, a0, a1, a2;
  if (kind === "low shelf" || kind === "high shelf") {
    const root = 2 * Math.sqrt(A) * alpha;
    const s = kind === "low shelf" ? 1 : -1;
    b0 = A * ((A + 1) - s * (A - 1) * cos + root);
    b1 = s * 2 * A * ((A - 1) - s * (A + 1) * cos);
    b2 = A * ((A + 1) - s * (A - 1) * cos - root);
    a0 = (A + 1) + s * (A - 1) * cos + root;
    a1 = -s * 2 * ((A - 1) + s * (A + 1) * cos);
    a2 = (A + 1) + s * (A - 1) * cos - root;
  } else if (kind === "notch") {
    [b0, b1, b2, a0, a1, a2] = [1, -2 * cos, 1, 1 + alpha, -2 * cos, 1 - alpha];
  } else if (kind === "band pass") {
    [b0, b1, b2, a0, a1, a2] = [alpha, 0, -alpha, 1 + alpha, -2 * cos, 1 - alpha];
  } else {
    [b0, b1, b2, a0, a1, a2] = [1 + alpha * A, -2 * cos, 1 - alpha * A, 1 + alpha / A, -2 * cos, 1 - alpha / A];
  }
  return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
}

/** A band's gain in dB at one frequency. */
function bandDb(band, hertz, rate) {
  if (!band.on) return 0;
  if (CUTS.includes(band.kind)) {
    const order = Math.max(1, Math.round(band.slope / 6));
    const ratio = Math.tan((Math.PI * Math.min(hertz, 0.499 * rate)) / rate) / Math.tan((Math.PI * Math.min(band.freq, 0.49 * rate)) / rate);
    const term = band.kind === "low cut" ? 1 / Math.max(ratio, 1e-12) : ratio;
    return -10 * Math.log10(1 + term ** (2 * order));
  }
  if (GAINED.includes(band.kind) && Math.abs(band.gain) < 1e-9) return 0;
  const [b0, b1, b2, a1, a2] = biquad(band.kind, band.freq, band.q, band.gain, rate);
  const w = (2 * Math.PI * Math.min(hertz, 0.499 * rate)) / rate;
  const c1 = Math.cos(w), s1 = Math.sin(w), c2 = Math.cos(2 * w), s2 = Math.sin(2 * w);
  const numRe = b0 + b1 * c1 + b2 * c2, numIm = -(b1 * s1 + b2 * s2);
  const denRe = 1 + a1 * c1 + a2 * c2, denIm = -(a1 * s1 + a2 * s2);
  const magnitude = Math.sqrt((numRe ** 2 + numIm ** 2) / Math.max(1e-30, denRe ** 2 + denIm ** 2));
  return 20 * Math.log10(Math.max(1e-12, magnitude));
}

function eqEditor(node) {
  let selected = -1;
  let held = null;
  let drag = null;
  const box = { width: 1, height: 1 };
  const rate = () => Number(node.__wasAudioPanel?.report()?.eq?.rate) || 48000;
  const high = () => Math.min(20000, rate() / 2);
  const text = () => String(findWidget(node, "bands")?.value ?? "");
  const current = () => (held ? { bands: held } : parseBands(text()));
  const yOf = (db) => box.height / 2 - (db / RANGE_DB) * (box.height / 2 - 10);
  const dbOf = (py) => ((box.height / 2 - py) / (box.height / 2 - 10)) * RANGE_DB;
  const dotOf = (band) => ({ px: logX(band.freq, box.width, high()), py: GAINED.includes(band.kind) ? yOf(band.gain) : yOf(0) });
  const hitAt = (at) => (current().bands ?? []).findIndex((band) => {
    const dot = dotOf(band);
    return Math.hypot(dot.px - at.x, dot.py - at.y) <= GRAB;
  });
  const commit = (bands) => node.__wasAudioPanel?.write("bands", writeBands(bands));

  const footer = el("div", "display:flex;align-items:center;gap:6px;flex-wrap:wrap;min-height:20px;font-size:11px");
  const kindSelect = el("select", `font:11px system-ui,sans-serif;background:${themeVar("inputBg")};`
    + `color:${themeVar("fg")};border:1px solid ${themeVar("border")};border-radius:4px;padding:1px 4px`);
  for (const kind of KINDS) {
    const option = el("option", "", kind);
    option.value = kind;
    kindSelect.append(option);
  }
  const readout = el("span", `color:${themeVar("fgMuted")};font-variant-numeric:tabular-nums`);
  const toggle = button("Bypass", () => {
    const { bands } = parseBands(text());
    if (!bands?.[selected]) return;
    bands[selected].on = !bands[selected].on;
    commit(bands);
  }, { kind: "tool", logName: EXT_NAME, hint: "Bypass or restore the selected band" });
  const remove = button("Remove", () => {
    const { bands } = parseBands(text());
    if (!bands?.[selected]) return;
    bands.splice(selected, 1);
    selected = -1;
    commit(bands);
  }, { kind: "tool", logName: EXT_NAME, hint: "Remove the selected band (Delete)" });
  for (const item of [toggle, remove]) item.style.cssText += ";padding:1px 8px;font-size:11px;line-height:16px;border-radius:4px";
  kindSelect.addEventListener("change", (event) => {
    event.stopPropagation();
    const { bands } = parseBands(text());
    if (!bands?.[selected]) return;
    const band = bands[selected];
    band.kind = kindSelect.value;
    if (CUTS.includes(band.kind)) band.slope = band.slope || 12;
    commit(bands);
  });
  footer.append(kindSelect, readout, toggle, remove);

  const paintFooter = () => {
    const { bands, error } = current();
    const band = bands?.[selected];
    const shown = Boolean(band);
    for (const item of [kindSelect, toggle, remove]) item.style.display = shown ? "" : "none";
    if (error) {
      readout.textContent = `bands text has an error: ${error}`;
      readout.style.color = themeVar("error");
      return;
    }
    readout.style.color = themeVar("fgMuted");
    if (!band) {
      readout.textContent = `${bands?.length ?? 0} bands: drag a dot, double-click to add, wheel for Q after a click`;
      return;
    }
    kindSelect.value = band.kind;
    toggle.textContent = band.on ? "Bypass" : "Restore";
    const hz = band.freq >= 1000 ? `${(band.freq / 1000).toFixed(2)} kHz` : `${band.freq.toFixed(0)} Hz`;
    readout.textContent = CUTS.includes(band.kind) ? `${hz} · ${band.slope} dB/oct`
      : `${hz} · ${band.gain >= 0 ? "+" : ""}${band.gain.toFixed(1)} dB · Q ${band.q.toFixed(2)}`;
  };

  const section = createGraphSection({
    height: 150,
    hint: "Drag a dot for frequency and gain; wheel for Q or slope after a click; double-click to add a band or bypass one; Delete removes",
    draw(ctx, size, theme) {
      box.width = size.width;
      box.height = size.height;
      const top = high();
      const spectrum = node.__wasAudioPanel?.report()?.eq?.spectrum;
      ctx.font = "9px system-ui,sans-serif";
      ctx.strokeStyle = theme.border;
      ctx.lineWidth = 1;
      for (const hz of [50, 100, 200, 500, 1000, 2000, 5000, 10000]) {
        if (hz >= top) continue;
        const px = Math.round(logX(hz, size.width, top)) + 0.5;
        ctx.globalAlpha = 0.35;
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, size.height);
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = theme.fgMuted;
        ctx.fillText(hz >= 1000 ? `${hz / 1000}k` : String(hz), px + 2, size.height - 3);
      }
      for (const db of [-18, -12, -6, 0, 6, 12, 18]) {
        ctx.globalAlpha = db === 0 ? 0.8 : 0.25;
        ctx.beginPath();
        ctx.moveTo(0, Math.round(yOf(db)) + 0.5);
        ctx.lineTo(size.width, Math.round(yOf(db)) + 0.5);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      if (spectrum?.freqs?.length) {
        const peak = Math.max(...spectrum.in, ...spectrum.out);
        const sy = (db) => size.height - ((db - (peak - 80)) / 80) * size.height;
        const trace = (values, fill) => {
          ctx.beginPath();
          spectrum.freqs.forEach((hz, index) => {
            const px = logX(hz, size.width, top);
            if (index === 0) ctx.moveTo(px, sy(values[index]));
            else ctx.lineTo(px, sy(values[index]));
          });
          if (fill) {
            ctx.lineTo(size.width, size.height);
            ctx.lineTo(0, size.height);
            ctx.closePath();
            ctx.fill();
          } else ctx.stroke();
        };
        ctx.fillStyle = theme.fgMuted;
        ctx.globalAlpha = 0.18;
        trace(spectrum.in, true);
        ctx.globalAlpha = 0.6;
        ctx.strokeStyle = theme.fgMuted;
        trace(spectrum.out, false);
        ctx.globalAlpha = 1;
      }
      const { bands, error } = current();
      if (error || !bands) return;
      ctx.strokeStyle = theme.accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let px = 0; px <= size.width; px += 2) {
        const hz = logHz(px, size.width, top);
        const total = bands.reduce((sum, band) => sum + bandDb(band, hz, rate()), 0);
        const py = yOf(Math.max(-RANGE_DB - 6, Math.min(RANGE_DB + 6, total)));
        if (px === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      const frame = { width: size.width, height: size.height, top: 0, theme };
      bands.forEach((band, index) => {
        const dot = dotOf(band);
        ctx.globalAlpha = band.on ? 1 : 0.4;
        ctx.fillStyle = index === selected ? theme.fg : theme.warning;
        ctx.beginPath();
        ctx.arc(dot.px, dot.py, index === selected ? 6 : 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = theme.bgDark ?? theme.bg;
        ctx.font = "bold 8px system-ui,sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(index + 1), dot.px, dot.py + 0.5);
        ctx.textAlign = "start";
        ctx.textBaseline = "alphabetic";
      });
      drawLabel(ctx, frame, "dB", 3, yOf(RANGE_DB - 3), { muted: true });
    },
    pointer: {
      cursor: (at) => (hitAt(at) >= 0 ? "grab" : "crosshair"),
      down(at) {
        const index = hitAt(at);
        selected = index;
        paintFooter();
        if (index < 0) return true;
        const parsed = parseBands(text());
        if (parsed.error) return false;
        held = parsed.bands;
        drag = { index };
        return true;
      },
      move(at, event) {
        if (!drag || !held) return;
        const band = held[drag.index];
        band.freq = Math.round(Math.max(LOG_LOW, Math.min(high(), logHz(at.x, box.width, high()))) * 10) / 10;
        if (GAINED.includes(band.kind)) {
          const gain = Math.max(-RANGE_DB, Math.min(RANGE_DB, dbOf(at.y)));
          band.gain = Math.round(gain * (event?.shiftKey ? 10 : 2)) / (event?.shiftKey ? 10 : 2);
        }
        paintFooter();
      },
      up() {
        if (drag && held) commit(held);
        drag = null;
        held = null;
        paintFooter();
      },
      doubleClick(at) {
        const parsed = parseBands(text());
        if (parsed.error) return false;
        const index = hitAt(at);
        if (index >= 0) {
          parsed.bands[index].on = !parsed.bands[index].on;
        } else {
          if (parsed.bands.length >= MAX_BANDS) return false;
          parsed.bands.push({
            kind: "bell", freq: Math.round(logHz(at.x, box.width, high())), on: true,
            gain: Math.round(Math.max(-RANGE_DB, Math.min(RANGE_DB, dbOf(at.y))) * 2) / 2, q: 1, slope: 12,
          });
          selected = parsed.bands.length - 1;
        }
        commit(parsed.bands);
        paintFooter();
        return true;
      },
    },
    wheel(event, at) {
      const parsed = parseBands(text());
      if (parsed.error) return false;
      const index = selected >= 0 ? selected : hitAt(at);
      const band = parsed.bands[index];
      if (!band) return false;
      if (CUTS.includes(band.kind)) band.slope = Math.max(6, Math.min(96, band.slope + (event.deltaY < 0 ? 6 : -6)));
      else band.q = Math.round(Math.max(0.1, Math.min(20, band.q * (event.deltaY < 0 ? 1.12 : 1 / 1.12))) * 100) / 100;
      selected = index;
      commit(parsed.bands);
      paintFooter();
      return true;
    },
    key(event) {
      if ((event.key === "Delete" || event.key === "Backspace") && selected >= 0) {
        const parsed = parseBands(text());
        if (parsed.error || !parsed.bands[selected]) return true;
        parsed.bands.splice(selected, 1);
        selected = -1;
        commit(parsed.bands);
        paintFooter();
        return true;
      }
      return event.key === "Delete" || event.key === "Backspace";
    },
  });

  const wrapper = el("div", "display:flex;flex-direction:column;gap:3px;flex:0 0 auto");
  wrapper.append(section.element, footer);
  paintFooter();
  return {
    element: wrapper,
    update: () => {
      paintFooter();
      section.repaint();
    },
    repaint: () => {
      paintFooter();
      section.repaint();
    },
    dispose: () => section.dispose(),
  };
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_audio_equalizer_wrapped) return;
    proto.__was_audio_equalizer_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const node = this;
        let editor = null;
        const built = mountAudioPanel(this, {
          initial: "out",
          height: 380,
          extra: () => {
            editor = eqEditor(node);
            return editor;
          },
        });
        if (built) {
          node.__wasAudioPanel = built.panel;
          chainWidgetCallback(node, "bands", () => editor?.repaint(), EXT_NAME);
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the equaliser:`, error);
      }
      return result;
    };
  },
});
