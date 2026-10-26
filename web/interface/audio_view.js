/**
 * The audio panel: a written side's waveform under a time ruler, a playhead, a before and after
 * toggle, a transport and a level meter, with an adopter's layers, gestures and controls.
 *
 * Times are seconds of the side shown.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { createPlayer } from "./audio_engine.js";
import { drawLabel } from "./canvas_label.js";
import { formatTime, holdAudio, keptAudio, releaseAudio, spanRange } from "./audio_media.js";
import { ICON, drawFidelityGlyph } from "./icons.js";
import { drawStandIn, loadPlaceholder, standInTitle } from "./placeholder.js";
import { captureWheel, elementPoint, wheelPixels } from "./pointer.js";
import { PREVIEW_STATE, connected, executionId, nodeLocator, ownOutput } from "./preview.js";
import { withGraphChange } from "./region.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onRunEnded } from "./run_events.js";
import { onThemeChange, readTheme, themeVar } from "./theme.js";
import { appendInterfaceWidget, graphLoading, keepOnNode } from "./widget.js";
import { button, el } from "./window_kit.js";

const LOG_NAME = "WASNodeSuite.AudioPanel";

// The setting that turns every audio panel on or off.
const SETTING_ID = "WAS.Audio.ShowPanels";

// The `ui` keys a node writes its sides and its figures under.
export const SIDES_KEY = "was_audio";
export const REPORT_KEY = "was_audio_report";

// Where the last sides and figures are kept on the node, so a reload draws them again.
const REMEMBERED_KEY = "was_audio_sides";

// The largest figures object kept on the node, in characters of JSON.
const REMEMBERED_LIMIT = 65536;

// Height the panel opens at, in node units, and the narrowest it is worth drawing in.
const PANEL_HEIGHT = 200;
const PANEL_MIN_WIDTH = 300;

// Height of the time ruler across the top of the waveform.
const RULER = 16;

// Shortest span the view zooms to, in seconds.
const MIN_SPAN = 0.002;

// Spacing a ruler label is given, in element pixels.
const LABEL_SPACING = 72;

// Ruler steps, in seconds.
const STEPS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60,
  120, 300, 600, 900, 1800, 3600];

// The level meter's range in dBFS, and how long a peak marker holds.
const METER_FLOOR = -48;
const PEAK_HOLD_MS = 1500;

// Pixels a press may move and still read as a click.
const CLICK_SLOP = 3;

const SMALL_BUTTON = "padding:1px 8px;font-size:11px;line-height:16px;border-radius:4px";

/**
 * A widget by its name, including a widget nested under a dynamic combo.
 *
 * @param {object} node - The node holding it.
 * @param {string} name - The widget's own name.
 * @returns {object|undefined} The widget.
 */
export function findWidget(node, name) {
  const widgets = node?.widgets ?? [];
  return widgets.find((widget) => widget?.name === name)
    ?? widgets.find((widget) => typeof widget?.name === "string" && widget.name.endsWith(`.${name}`));
}

/**
 * Write one widget as a single undoable edit.
 *
 * @param {object} node - The node holding the widget.
 * @param {string} name - The widget's name.
 * @param {*} value - The value to write.
 * @returns {boolean} True when the widget exists and took a new value.
 */
export function writeWidget(node, name, value) {
  const widget = findWidget(node, name);
  if (!widget || widget.value === value) return false;
  withGraphChange(() => {
    widget.value = value;
    widget.callback?.(value);
  });
  node.setDirtyCanvas?.(true, true);
  return true;
}

/**
 * A number widget's value.
 *
 * @param {object} node - The node holding the widget.
 * @param {string} name - The widget's name.
 * @param {number} [fallback] - Answered where there is no such widget or no number in it.
 * @returns {number} The value.
 */
export function readNumber(node, name, fallback = 0) {
  const value = Number(findWidget(node, name)?.value);
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Seconds rounded to the thousandth the widgets hold.
 *
 * @param {number} seconds - A time.
 * @returns {number} The time to three places.
 */
export function roundTime(seconds) {
  return Math.round(Math.max(0, seconds) * 1000) / 1000;
}

/**
 * A level in dBFS as the panel prints it.
 *
 * @param {number|null} db - The level, null for silence.
 * @returns {string} `-6.0 dBFS`, or `silent`.
 */
export function formatLevel(db) {
  return typeof db === "number" && Number.isFinite(db) ? `${db.toFixed(1)} dBFS` : "silent";
}

/**
 * Build the audio panel for one node.
 *
 * @param {object} node - The node the panel belongs to.
 * @param {object} [options] - How the panel reads, draws and plays.
 * @param {Array<{key: string, label: string}>} [options.sides] - The sides a node writes, in toggle order.
 * @param {string} [options.initial] - The side shown first.
 * @param {Array<Function>} [options.layers] - `(ctx, frame, panel)` callbacks drawn over the waveform.
 * @param {(panel: object) => object} [options.tool] - Builds the gestures an adopter takes.
 * @param {(report: object, side: string, panel: object) => string} [options.figures] - The header line.
 * @param {(bar: HTMLElement, panel: object) => void} [options.controls] - Adds transport controls.
 * @param {(panel: object) => Array<object>|null} [options.audition] - Pieces Play sounds instead of
 *   the side shown, or null to play the side.
 * @param {(seconds: number, from: string, to: string, report: object) => number} [options.mapTime] -
 *   Where a time on one side falls on the other, for the toggle.
 * @param {(panel: object) => {element: HTMLElement, update: (report: object) => void}} [options.extra] -
 *   Builds a section drawn between the waveform and the transport, told each new report.
 * @param {() => void} [options.onPlay] - Called whenever playback starts.
 * @param {boolean} [options.select] - A drag on the waveform makes a selection rather than seeking.
 * @param {number} [options.height] - Height the panel opens at.
 * @returns {object} The element, its height, its minimum width, `dispose`, `refresh` and `panel`.
 */
export function createAudioPanel(node, options = {}) {
  const sides = options.sides ?? [{ key: "in", label: "Before" }, { key: "out", label: "After" }];
  const layers = options.layers ?? [];
  const root = el("div", "position:relative;width:100%;height:100%;display:flex;flex-direction:column;"
    + `gap:4px;padding:4px;box-sizing:border-box;border-radius:4px;overflow:hidden;outline:none;`
    + `background:${themeVar("panelBg")};color:${themeVar("fg")};font:11px system-ui,sans-serif`);
  root.tabIndex = 0;

  const header = el("div", "display:flex;align-items:center;gap:6px;flex:0 0 auto;min-height:18px");
  const toggle = el("div", `display:flex;flex:0 0 auto;border:1px solid ${themeVar("border")};`
    + "border-radius:4px;overflow:hidden");
  const figuresLine = el("div", `flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;`
    + `white-space:nowrap;color:${themeVar("fgMuted")};font-variant-numeric:tabular-nums;text-align:right`);
  const glyph = el("canvas", "flex:0 0 auto;width:11px;height:11px");
  header.append(toggle, figuresLine, glyph);

  const stage = el("div", `position:relative;flex:1 1 auto;min-height:60px;border-radius:3px;`
    + `overflow:hidden;background:${themeVar("bgDark")}`);
  const canvas = el("canvas", "position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none");
  stage.append(canvas);
  const message = el("div", "position:absolute;left:0;right:0;bottom:6px;text-align:center;pointer-events:none;"
    + `color:${themeVar("error")};display:none`);
  stage.append(message);

  const bar = el("div", "display:flex;align-items:center;gap:5px;flex:0 0 auto;flex-wrap:wrap");
  const playButton = button("Play", () => togglePlay(), { kind: "tool", logName: LOG_NAME, hint: "Play or stop (Space)" });
  playButton.style.cssText += `;${SMALL_BUTTON};min-width:44px`;
  const loopButton = button("Loop", () => {
    looping = !looping;
    paintLoop();
    if (player.playing()) startPlaying(player.position());
  }, { kind: "tool", logName: LOG_NAME, hint: "Loop the selection, or the whole clip" });
  loopButton.style.cssText += `;${SMALL_BUTTON}`;
  const fitButton = button("Fit", () => fit(), { kind: "tool", logName: LOG_NAME, hint: "Show the whole clip (F)" });
  fitButton.style.cssText += `;${SMALL_BUTTON}`;
  const clock = el("span", "flex:0 0 auto;font-variant-numeric:tabular-nums;white-space:nowrap", "0:00.000");
  const spacer = el("div", "flex:1 1 auto");
  const meter = el("canvas", "flex:0 0 auto;width:84px;height:14px");
  meter.title = "Left and right level";
  bar.append(playButton, loopButton, fitButton, clock, spacer, meter);
  root.append(header, stage, bar);

  const context = canvas.getContext("2d");
  const meterContext = meter.getContext("2d");

  let entries = {};
  let clips = {};
  let holds = {};
  let report = {};
  let sideKey = options.initial ?? sides[sides.length - 1].key;
  let status = connected() ? PREVIEW_STATE.WAITING : PREVIEW_STATE.CONNECTING;
  let view = { t0: 0, t1: 1 };
  let fitted = true;
  let playhead = 0;
  let hover = null;
  let selection = null;
  let looping = false;
  let loaded = "";
  let turn = 0;
  let frame = 0;
  let theme = readTheme();
  let gesture = null;
  let meterPeaks = [];
  let extra = null;

  const player = createPlayer({
    onState: (playing) => {
      playButton.textContent = playing ? "Stop" : "Play";
      if (playing) {
        try {
          options.onPlay?.();
        } catch (error) {
          console.error(`[${LOG_NAME}] A play handler failed:`, error);
        }
      }
      if (playing) animate();
      else {
        playhead = player.position();
        schedule();
      }
    },
  });

  const clip = () => clips[sideKey] ?? null;
  const duration = () => clip()?.duration ?? 0;
  const width = () => canvas.clientWidth || 1;
  const xAt = (seconds) => ((seconds - view.t0) / Math.max(1e-9, view.t1 - view.t0)) * width();
  const timeAt = (x) => view.t0 + (x / width()) * (view.t1 - view.t0);

  /** The lanes the waveform is drawn in, one per channel, in element pixels. */
  const lanes = () => {
    const count = Math.max(1, clip()?.channels.length ?? 1);
    const top = RULER + 1;
    const height = Math.max(1, canvas.clientHeight - top - 1);
    const each = height / count;
    return Array.from({ length: count }, (_, index) => ({ top: top + index * each, height: each }));
  };

  const panel = {
    node,
    root,
    canvas,
    player,
    side: () => sideKey,
    clip,
    clipOf: (key) => clips[key] ?? null,
    report: () => report,
    view: () => ({ ...view }),
    xAt,
    timeAt,
    lanes,
    repaint: () => {
      paintFigures();
      schedule();
    },
    playhead: () => playhead,
    seek: (seconds) => seek(seconds),
    selection: () => (selection ? [...selection] : null),
    setSelection: (range) => {
      selection = range && range[1] - range[0] > 0 ? [Math.max(0, range[0]), Math.min(duration() || range[1], range[1])] : null;
      schedule();
    },
    widget: (name) => findWidget(node, name),
    write: (name, value) => writeWidget(node, name, value),
    theme: () => theme,
    showSide: (key) => showSide(key),
  };
  const tool = options.tool?.(panel) ?? null;
  panel.tool = tool;

  // ---- drawing -------------------------------------------------------------------------------

  const tickStep = (span) => {
    const perPixel = span / width();
    return STEPS.find((step) => step / perPixel >= LABEL_SPACING) ?? STEPS[STEPS.length - 1];
  };

  function drawRuler(ctx, w) {
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, w, RULER);
    ctx.fillStyle = theme.border;
    ctx.fillRect(0, RULER, w, 1);
    const span = view.t1 - view.t0;
    const step = tickStep(span);
    const places = step < 0.01 ? 3 : step < 0.1 ? 2 : step < 1 ? 1 : 0;
    ctx.font = "9px system-ui,sans-serif";
    ctx.textBaseline = "top";
    const first = Math.floor(view.t0 / step) * step;
    for (let time = first; time <= view.t1 + step; time += step / 2) {
      const x = Math.round(xAt(time)) + 0.5;
      if (x < 0 || x > w) continue;
      const major = Math.abs(time / step - Math.round(time / step)) < 1e-6;
      ctx.fillStyle = major ? theme.fgMuted : theme.border;
      ctx.fillRect(x, major ? RULER - 6 : RULER - 3, 1, major ? 6 : 3);
      if (major) {
        ctx.fillStyle = theme.fgMuted;
        ctx.fillText(formatTime(Math.max(0, time), places), x + 3, 2);
      }
    }
  }

  function drawWave(ctx, w) {
    const held = clip();
    const samplesPerPixel = ((view.t1 - view.t0) * held.rate) / w;
    lanes().forEach((lane, channel) => {
      const middle = lane.top + lane.height / 2;
      const scale = (lane.height / 2) * 0.92;
      ctx.fillStyle = theme.border;
      ctx.fillRect(0, Math.round(middle), w, 1);
      if (channel > 0) {
        ctx.fillStyle = theme.borderLight ?? theme.border;
        ctx.fillRect(0, Math.round(lane.top), w, 1);
      }
      ctx.fillStyle = theme.accent;
      if (samplesPerPixel < 1.5) {
        const samples = held.channels[channel];
        const first = Math.max(0, Math.floor(view.t0 * held.rate) - 1);
        const last = Math.min(held.length, Math.ceil(view.t1 * held.rate) + 1);
        ctx.strokeStyle = theme.accent;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        for (let index = first; index < last; index += 1) {
          const x = xAt(index / held.rate);
          const y = middle - Math.max(-1.05, Math.min(1.05, samples[index])) * scale;
          if (index === first) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          if (samplesPerPixel < 0.25) ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
        }
        ctx.stroke();
        return;
      }
      for (let x = 0; x < w; x += 1) {
        const from = timeAt(x) * held.rate;
        const to = timeAt(x + 1) * held.rate;
        if (to <= 0 || from >= held.length) continue;
        const [low, high] = spanRange(held, channel, from, to);
        const top = middle - Math.min(1.05, high) * scale;
        const bottom = middle - Math.max(-1.05, low) * scale;
        ctx.fillStyle = theme.accent;
        ctx.fillRect(x, top, 1, Math.max(1, bottom - top));
        if (high >= 1 || low <= -1) {
          ctx.fillStyle = theme.error;
          if (high >= 1) ctx.fillRect(x, lane.top + 1, 1, 3);
          if (low <= -1) ctx.fillRect(x, lane.top + lane.height - 4, 1, 3);
        }
      }
    });
    if (clip() && view.t1 > held.duration) {
      const end = xAt(held.duration);
      ctx.fillStyle = theme.bg;
      ctx.globalAlpha = 0.6;
      ctx.fillRect(end, RULER + 1, w - end, canvas.clientHeight);
      ctx.globalAlpha = 1;
    }
  }

  function drawSelection(ctx, h) {
    if (!selection) return;
    const left = xAt(selection[0]);
    const right = xAt(selection[1]);
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = theme.fg;
    ctx.fillRect(left, RULER + 1, right - left, h - RULER - 1);
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = theme.fg;
    ctx.fillRect(Math.round(left), RULER + 1, 1, h - RULER - 1);
    ctx.fillRect(Math.round(right), RULER + 1, 1, h - RULER - 1);
    ctx.restore();
  }

  function draw() {
    frame = 0;
    if (!context) return;
    const ratio = surfaceRatio(canvas);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!(w > 0) || !(h > 0)) return;
    const pixelWidth = Math.max(1, Math.round(w * ratio));
    const pixelHeight = Math.max(1, Math.round(h * ratio));
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const ctx = context;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, pixelWidth, pixelHeight);
    ctx.scale(pixelWidth / w, pixelHeight / h);
    ctx.fillStyle = theme.bgDark ?? theme.bg;
    ctx.fillRect(0, 0, w, h);
    const held = clip();
    if (!held) {
      if (!drawStandIn(ctx, { x: 8, y: 8, w: w - 16, h: h - 16 })) {
        ctx.fillStyle = theme.fgMuted;
        ctx.font = "11px system-ui,sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Run to hear it", w / 2, h / 2);
        ctx.textAlign = "start";
      }
      drawFidelityGlyph(glyph, { icon: ICON.WARNING, detail: "" });
      glyph.title = standInTitle(status);
      return;
    }
    drawRuler(ctx, w);
    drawWave(ctx, w);
    for (const layer of layers) {
      ctx.save();
      try {
        layer(ctx, { width: w, height: h, top: RULER + 1, lanes: lanes(), xAt, timeAt, theme, view: { ...view } }, panel);
      } catch (error) {
        console.error(`[${LOG_NAME}] A layer failed to draw:`, error);
      }
      ctx.restore();
    }
    drawSelection(ctx, h);
    if (hover !== null && !gesture) {
      ctx.fillStyle = theme.fgMuted;
      ctx.globalAlpha = 0.5;
      ctx.fillRect(Math.round(xAt(hover)), RULER + 1, 1, h - RULER - 1);
      ctx.globalAlpha = 1;
    }
    const head = Math.round(xAt(playhead)) + 0.5;
    if (head >= 0 && head <= w) {
      ctx.fillStyle = theme.fg;
      ctx.fillRect(head - 0.5, 0, 1.5, h);
      ctx.beginPath();
      ctx.moveTo(head - 4, 0);
      ctx.lineTo(head + 4, 0);
      ctx.lineTo(head, 5);
      ctx.closePath();
      ctx.fill();
    }
    drawFidelityGlyph(glyph, held.exact
      ? { icon: ICON.EXACT, detail: held.detail }
      : { icon: ICON.APPROXIMATE, detail: held.detail });
    paintClock();
  }

  function drawMeter() {
    if (!meterContext) return;
    const ratio = surfaceRatio(meter);
    const w = meter.clientWidth || 84;
    const h = meter.clientHeight || 14;
    const pixelWidth = Math.max(1, Math.round(w * ratio));
    const pixelHeight = Math.max(1, Math.round(h * ratio));
    if (meter.width !== pixelWidth || meter.height !== pixelHeight) {
      meter.width = pixelWidth;
      meter.height = pixelHeight;
    }
    const ctx = meterContext;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, pixelWidth, pixelHeight);
    ctx.scale(pixelWidth / w, pixelHeight / h);
    const levels = player.levels();
    const now = performance.now();
    const shown = levels.length === 1 ? [levels[0], levels[0]] : levels.slice(0, 2);
    const at = (db) => Math.max(0, Math.min(1, (db - METER_FLOOR) / -METER_FLOOR)) * w;
    for (let row = 0; row < 2; row += 1) {
      const top = row * (h / 2) + 1;
      const tall = h / 2 - 2;
      ctx.fillStyle = theme.bgDark ?? theme.bg;
      ctx.fillRect(0, top, w, tall);
      const level = shown[row] ?? { peak: 0, rms: 0 };
      const db = level.peak > 0 ? 20 * Math.log10(level.peak) : -Infinity;
      const reach = at(db);
      const bands = [[METER_FLOOR, -12, theme.success], [-12, -3, theme.warning], [-3, 0, theme.error]];
      for (const [low, high, colour] of bands) {
        const left = at(low);
        const right = Math.min(reach, at(high));
        if (right > left) {
          ctx.fillStyle = colour;
          ctx.fillRect(left, top, right - left, tall);
        }
      }
      const kept = meterPeaks[row];
      if (!kept || db >= kept.db || now - kept.at > PEAK_HOLD_MS) meterPeaks[row] = { db, at: now };
      const peak = meterPeaks[row];
      if (Number.isFinite(peak.db) && peak.db > METER_FLOOR) {
        ctx.fillStyle = peak.db >= 0 ? theme.error : theme.fg;
        ctx.fillRect(Math.min(w - 1, at(peak.db)), top, 1, tall);
      }
    }
  }

  function paintClock() {
    const shownTime = hover !== null && !player.playing() ? hover : playhead;
    clock.textContent = `${formatTime(shownTime)} / ${formatTime(duration())}`;
  }

  function paintLoop() {
    loopButton.style.borderColor = looping ? themeVar("accent") : themeVar("border");
    loopButton.style.color = looping ? themeVar("accent") : themeVar("fg");
  }

  function schedule() {
    if (!frame) frame = requestAnimationFrame(draw);
  }

  function animate() {
    const step = () => {
      if (!player.playing()) {
        meterPeaks = [];
        drawMeter();
        draw();
        return;
      }
      playhead = player.position();
      const span = view.t1 - view.t0;
      if (playhead > view.t1 || playhead < view.t0) {
        view = { t0: Math.max(0, playhead - span * 0.1), t1: Math.max(0, playhead - span * 0.1) + span };
        fitted = false;
      }
      draw();
      drawMeter();
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // ---- view, sides and playback ---------------------------------------------------------------

  function fit() {
    const total = duration();
    view = { t0: 0, t1: total > 0 ? total : 1 };
    fitted = true;
    schedule();
  }

  function seek(seconds) {
    playhead = Math.max(0, Math.min(duration(), seconds));
    player.seek(playhead);
    if (player.playing()) startPlaying(playhead);
    schedule();
  }

  function startPlaying(from) {
    const held = clip();
    if (!held) return;
    const pieces = options.audition?.(panel);
    if (pieces?.length) {
      player.playPieces(held, pieces);
      return;
    }
    const range = looping ? (selection ?? [0, held.duration]) : null;
    player.play(held, { from, loop: range });
  }

  function togglePlay() {
    if (player.playing()) {
      player.stop();
      return;
    }
    if (!clip()) return;
    const start = selection && !looping && (playhead < selection[0] || playhead >= selection[1])
      ? selection[0] : playhead;
    startPlaying(start);
  }

  function paintToggle() {
    toggle.replaceChildren();
    const present = sides.filter((side) => clips[side.key] || entries[side.key]);
    toggle.style.display = present.length > 1 ? "flex" : "none";
    for (const side of present) {
      const chosen = side.key === sideKey;
      const item = el("button", `border:none;padding:1px 9px;font:11px system-ui,sans-serif;cursor:pointer;`
        + `background:${chosen ? themeVar("accent") : "transparent"};`
        + `color:${chosen ? themeVar("selectionText") : themeVar("fg")}`, side.label);
      item.type = "button";
      item.title = `Show and play ${side.label.toLowerCase()}`;
      item.addEventListener("click", (event) => {
        event.stopPropagation();
        showSide(side.key);
      });
      toggle.append(item);
    }
  }

  function showSide(key) {
    if (key === sideKey || !clips[key]) return;
    const wasPlaying = player.playing();
    const at = wasPlaying ? player.position() : playhead;
    const mapped = options.mapTime ? options.mapTime(at, sideKey, key, report) : at;
    const before = duration();
    sideKey = key;
    playhead = Math.max(0, Math.min(duration(), mapped));
    if (fitted || Math.abs(before - duration()) > 1e-6) fit();
    paintToggle();
    paintFigures();
    if (wasPlaying) startPlaying(playhead);
    schedule();
  }

  function paintFigures() {
    try {
      extra?.update?.(report);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to update the panel's section:`, error);
    }
    try {
      const text = options.figures?.(report, sideKey, panel);
      if (typeof text === "string") {
        figuresLine.textContent = text;
        return;
      }
      const facts = report?.[sideKey];
      if (!facts) {
        figuresLine.textContent = "";
        return;
      }
      const layout = facts.channels === 1 ? "mono" : facts.channels === 2 ? "stereo" : `${facts.channels} ch`;
      const batch = facts.batch > 1 ? ` · clip 1 of ${facts.batch}` : "";
      figuresLine.textContent = `${(facts.rate / 1000).toFixed(facts.rate % 1000 ? 1 : 0)} kHz · ${layout}`
        + ` · peak ${formatLevel(facts.peak_db)}${batch}`;
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to write the figures:`, error);
    }
  }

  // ---- reading what the node wrote ------------------------------------------------------------

  const sidesOf = (output) => {
    const list = Array.isArray(output?.[SIDES_KEY]) ? output[SIDES_KEY] : [];
    const found = {};
    for (const entry of list) {
      if (entry && typeof entry.side === "string" && ownOutput(entry)) found[entry.side] = entry;
    }
    const figures = Array.isArray(output?.[REPORT_KEY]) ? output[REPORT_KEY][0] : null;
    return { sides: found, report: figures && typeof figures === "object" ? figures : {} };
  };

  function show(found, remembered = false) {
    const sides = Object.fromEntries(Object.entries(found?.sides ?? {}).filter(([, entry]) => keptAudio(entry)));
    const figures = found?.report && typeof found.report === "object" ? found.report : {};
    const read = { sides, report: JSON.stringify(figures).length > REMEMBERED_LIMIT ? {} : figures };
    const named = Object.keys(sides);
    if (!named.length) return false;
    const signature = named.sort().map((key) => `${key}=${read.sides[key].filename}`).join("|");
    // A remembered report is shown once its clip has loaded.
    report = remembered && !(signature === loaded && Object.keys(clips).length) ? {} : read.report;
    paintFigures();
    if (signature === loaded) {
      schedule();
      return true;
    }
    loaded = signature;
    const mine = ++turn;
    status = PREVIEW_STATE.LOADING;
    const asked = named.map((key) => [key, holdAudio(read.sides[key])]);
    Promise.all(asked.map(([, hold]) => hold.promise)).then((results) => {
      if (mine !== turn) {
        asked.forEach(([, hold]) => releaseAudio(hold.address));
        return;
      }
      player.stop();
      Object.values(holds).forEach((address) => releaseAudio(address));
      holds = {};
      clips = {};
      entries = read.sides;
      asked.forEach(([key, hold], index) => {
        holds[key] = hold.address;
        if (results[index]) clips[key] = results[index];
      });
      const any = Object.keys(clips);
      // A remembered side that is gone reads as not yet run; a fresh one that fails is a fault.
      const failed = !any.length && !remembered;
      if (remembered) report = any.length ? read.report : {};
      status = any.length ? PREVIEW_STATE.READY : failed ? PREVIEW_STATE.FAILED : PREVIEW_STATE.WAITING;
      message.style.display = failed ? "block" : "none";
      message.textContent = failed ? "Audio could not load" : "";
      if (!any.length) loaded = "";
      if (!clips[sideKey]) sideKey = clips[options.initial] ? options.initial : any[any.length - 1] ?? sideKey;
      playhead = Math.min(playhead, duration());
      if (selection && selection[1] > duration()) selection = null;
      fit();
      paintToggle();
      paintFigures();
      tool?.loaded?.();
    });
    return true;
  }

  function remember(read) {
    let kept = { sides: read.sides, report: read.report };
    if (JSON.stringify(kept).length > REMEMBERED_LIMIT) kept = { sides: read.sides, report: {} };
    if (JSON.stringify(node.properties?.[REMEMBERED_KEY]) === JSON.stringify(kept)) return;
    keepOnNode(node, REMEMBERED_KEY, kept);
  }

  function refresh() {
    try {
      const store = app?.nodeOutputs ?? {};
      const output = store[nodeLocator(node)] ?? store[executionId(node)] ?? store[String(node.id)];
      const read = sidesOf(output);
      if (show(read)) {
        remember(read);
        return;
      }
      const kept = node.properties?.[REMEMBERED_KEY];
      if (kept && show({ sides: kept.sides ?? {}, report: kept.report ?? {} }, true)) return;
      if (!Object.keys(clips).length) {
        status = connected() ? PREVIEW_STATE.WAITING : PREVIEW_STATE.CONNECTING;
        schedule();
      }
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the node's audio:`, error);
    }
  }

  const onExecuted = (event) => {
    try {
      const detail = event?.detail;
      const shown = String(detail?.display_node ?? detail?.node ?? "");
      if (!shown || shown !== executionId(node)) return;
      const read = sidesOf(detail?.output);
      if (show(read)) remember(read);
      else {
        report = read.report ?? report;
        paintFigures();
        schedule();
      }
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the node's report:`, error);
    }
  };
  api.addEventListener("executed", onExecuted);
  const onReconnected = () => {
    if (!loaded) refresh();
  };
  api.addEventListener("reconnected", onReconnected);

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const configured = originalOnConfigure?.apply(this, args);
    if (!graphLoading()) refresh();
    return configured;
  };

  // ---- gestures -------------------------------------------------------------------------------

  const pointAt = (event) => {
    const at = elementPoint(canvas, event);
    return { ...at, time: Math.max(0, timeAt(at.x)) };
  };

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    event.stopPropagation();
    root.focus({ preventScroll: true });
    if (!clip()) return;
    const at = pointAt(event);
    if (tool?.pointerDown?.(event, at)) {
      gesture = { kind: "tool", start: at };
    } else if (at.y <= RULER) {
      gesture = { kind: "pan", start: at, view: { ...view } };
    } else {
      gesture = { kind: options.select ? "select" : "seek", start: at, moved: false };
      if (!options.select) seek(at.time);
    }
    try {
      canvas.setPointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${LOG_NAME}] The pointer could not be captured:`, error);
    }
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!clip()) return;
    const at = pointAt(event);
    if (gesture && event.buttons === 0) {
      endGesture(event, at);
      return;
    }
    if (!gesture) {
      hover = at.y > RULER ? at.time : null;
      canvas.style.cursor = tool?.cursor?.(at) || (at.y <= RULER ? "grab" : options.select ? "text" : "pointer");
      paintClock();
      schedule();
      return;
    }
    event.stopPropagation();
    const moved = Math.abs(at.x - gesture.start.x) > CLICK_SLOP;
    if (gesture.kind === "tool") tool?.pointerMove?.(event, at);
    else if (gesture.kind === "pan") {
      const shift = ((gesture.start.x - at.x) / width()) * (gesture.view.t1 - gesture.view.t0);
      const span = gesture.view.t1 - gesture.view.t0;
      const start = Math.max(0, Math.min(Math.max(0, duration() - span), gesture.view.t0 + shift));
      view = { t0: start, t1: start + span };
      fitted = false;
    } else if (gesture.kind === "select" && (moved || gesture.moved)) {
      gesture.moved = true;
      const a = Math.min(gesture.start.time, at.time);
      const b = Math.max(gesture.start.time, at.time);
      selection = [Math.max(0, a), Math.min(duration(), b)];
      tool?.selectionChanged?.();
    } else if (gesture.kind === "seek" && moved) {
      seek(at.time);
    }
    schedule();
  });

  function endGesture(event, at) {
    const ended = gesture;
    gesture = null;
    if (!ended) return;
    if (ended.kind === "tool") tool?.pointerUp?.(event, at ?? ended.start);
    else if (ended.kind === "select" && !ended.moved) {
      selection = null;
      seek((at ?? ended.start).time);
      tool?.selectionChanged?.();
    }
    try {
      canvas.releasePointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${LOG_NAME}] The pointer was already released:`, error);
    }
    schedule();
  }

  canvas.addEventListener("pointerup", (event) => {
    event.stopPropagation();
    endGesture(event, clip() ? pointAt(event) : null);
  });
  canvas.addEventListener("pointercancel", (event) => endGesture(event, null));
  canvas.addEventListener("lostpointercapture", (event) => {
    if (gesture) endGesture(event, null);
  });
  canvas.addEventListener("pointerleave", () => {
    hover = null;
    paintClock();
    schedule();
  });
  canvas.addEventListener("dblclick", (event) => {
    event.stopPropagation();
    if (!clip()) return;
    if (tool?.doubleClick?.(event, pointAt(event))) return;
    fit();
  });
  root.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });

  const releaseWheel = captureWheel(root, (event) => {
    const held = clip();
    if (!held || !stage.contains(event.target)) return false;
    const distance = wheelPixels(event, stage);
    const span = view.t1 - view.t0;
    const before = { ...view };
    if (event.shiftKey || Math.abs(distance.x) > Math.abs(distance.y)) {
      const shift = ((distance.x || distance.y) / width()) * span;
      const start = Math.max(0, Math.min(Math.max(0, held.duration - span), view.t0 + shift));
      view = { t0: start, t1: start + span };
    } else {
      const at = timeAt(elementPoint(canvas, event).x);
      const next = Math.max(MIN_SPAN, Math.min(Math.max(held.duration, MIN_SPAN), span * Math.exp(distance.y * 0.002)));
      const share = (at - view.t0) / span;
      let start = at - share * next;
      start = Math.max(0, Math.min(Math.max(0, held.duration - next), start));
      view = { t0: start, t1: start + next };
    }
    const moved = before.t0 !== view.t0 || before.t1 !== view.t1;
    if (moved) {
      fitted = view.t0 <= 0 && view.t1 >= held.duration;
      schedule();
    }
    return moved;
  });

  root.addEventListener("keydown", (event) => {
    if (event.target !== root) return;
    let used = false;
    try {
      used = tool?.key?.(event) === true;
    } catch (error) {
      console.error(`[${LOG_NAME}] A key handler failed:`, error);
    }
    if (!used) {
      if (event.key === " ") {
        togglePlay();
        used = true;
      } else if (event.key === "Home") {
        seek(0);
        used = true;
      } else if (event.key === "End") {
        seek(duration());
        used = true;
      } else if (event.key === "f" || event.key === "F") {
        fit();
        used = true;
      } else if (event.key === "Escape" && selection) {
        selection = null;
        tool?.selectionChanged?.();
        schedule();
        used = true;
      }
    }
    if (used) {
      event.preventDefault();
      event.stopPropagation();
    }
  });

  const resized = new ResizeObserver(() => {
    schedule();
    drawMeter();
  });
  resized.observe(stage);
  const stopRatio = watchSurfaceRatio(canvas, () => {
    schedule();
    drawMeter();
  });
  const stopTheme = onThemeChange((next) => {
    theme = next;
    paintLoop();
    schedule();
    drawMeter();
  });
  const stopRun = onRunEnded(() => refresh());
  loadPlaceholder().then(() => schedule());

  options.controls?.(bar, panel);
  bar.append(spacer, meter);
  extra = options.extra?.(panel) ?? null;
  if (extra?.element) root.insertBefore(extra.element, bar);
  paintToggle();
  paintLoop();
  refresh();
  schedule();

  const dispose = () => {
    try {
      turn += 1;
      stopRun();
      stopTheme();
      stopRatio();
      releaseWheel();
      resized.disconnect();
      api.removeEventListener?.("executed", onExecuted);
      api.removeEventListener?.("reconnected", onReconnected);
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      player.dispose();
      Object.values(holds).forEach((address) => releaseAudio(address));
      holds = {};
      clips = {};
      tool?.dispose?.();
      extra?.dispose?.();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to release the panel:`, error);
    }
  };

  return {
    element: root,
    height: options.height ?? PANEL_HEIGHT,
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth: PANEL_MIN_WIDTH,
    dispose,
    refresh,
    panel,
  };
}

/**
 * Whether audio panels are drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
export function audioPanelsEnabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID);
    return typeof legacy === "boolean" ? legacy : true;
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to read ${SETTING_ID}:`, error);
    return true;
  }
}

/**
 * Build an audio panel and append it to a node, released when the node is removed.
 *
 * @param {object} node - The node being created.
 * @param {object} options - What `createAudioPanel` takes.
 * @param {{name: string, type: string}} [names] - The widget's name and type.
 * @returns {object|null} What `createAudioPanel` answered, or null while panels are off.
 */
export function mountAudioPanel(node, options, names = { name: "was_audio_ui", type: "was_audio_panel" }) {
  if (!audioPanelsEnabled()) return null;
  const built = createAudioPanel(node, options);
  appendInterfaceWidget(node, built, names);
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const removed = originalOnRemoved?.apply(this, args);
    built.dispose();
    return removed;
  };
  return built;
}

/**
 * A fade curve's gain, the same shapes the nodes fade on.
 *
 * @param {string} curve - `equal power`, `linear`, `s-curve`, `exponential` or `logarithmic`.
 * @param {number} progress - 0 at the fade's start to 1 at its end.
 * @returns {number} Gain, 0 to 1, rising.
 */
export function curveGain(curve, progress) {
  const p = Math.max(0, Math.min(1, progress));
  if (curve === "linear") return p;
  if (curve === "s-curve") return 0.5 - 0.5 * Math.cos(Math.PI * p);
  if (curve === "exponential") return p ** 3;
  if (curve === "logarithmic") return 1 - (1 - p) ** 3;
  return Math.sin(0.5 * Math.PI * p);
}

/**
 * Draw vertical markers at times.
 *
 * @param {CanvasRenderingContext2D} ctx - The panel's context.
 * @param {object} frame - The frame a layer is handed.
 * @param {number[]} times - Seconds.
 * @param {string} colour - Line colour.
 * @param {{labels?: boolean, dashed?: boolean}} [style] - Number each marker, or dash it.
 * @returns {void}
 */
export function drawMarkers(ctx, frame, times, colour, style = {}) {
  ctx.fillStyle = colour;
  ctx.strokeStyle = colour;
  ctx.font = "9px system-ui,sans-serif";
  ctx.textBaseline = "top";
  (times ?? []).forEach((time, index) => {
    const x = Math.round(frame.xAt(time)) + 0.5;
    if (x < -1 || x > frame.width + 1) return;
    ctx.beginPath();
    if (style.dashed) ctx.setLineDash([3, 3]);
    ctx.moveTo(x, frame.top);
    ctx.lineTo(x, frame.height);
    ctx.stroke();
    ctx.setLineDash([]);
    if (style.labels) drawLabel(ctx, frame, String(index + 1), x + 3, frame.top + 9);
  });
}

/**
 * Shade a span of time.
 *
 * @param {CanvasRenderingContext2D} ctx - The panel's context.
 * @param {object} frame - The frame a layer is handed.
 * @param {number} from - Start, seconds.
 * @param {number} to - End, seconds.
 * @param {string} colour - Fill colour.
 * @param {number} [alpha] - Opacity.
 * @returns {void}
 */
export function shadeSpan(ctx, frame, from, to, colour, alpha = 0.25) {
  const left = Math.max(0, frame.xAt(from));
  const right = Math.min(frame.width, frame.xAt(to));
  if (right <= left) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = colour;
  ctx.fillRect(left, frame.top, right - left, frame.height - frame.top);
  ctx.restore();
}

/**
 * Draw a gain envelope across the lanes, full gain at the top of each lane.
 *
 * @param {CanvasRenderingContext2D} ctx - The panel's context.
 * @param {object} frame - The frame a layer is handed.
 * @param {(seconds: number) => number} gainAt - Gain, 0 to 1, at a time.
 * @param {string} colour - Line colour.
 * @param {[number, number]} [span] - Times the envelope is drawn over; the whole view by default.
 * @returns {void}
 */
export function drawEnvelope(ctx, frame, gainAt, colour, span = null) {
  const from = span ? Math.max(0, frame.xAt(span[0])) : 0;
  const to = span ? Math.min(frame.width, frame.xAt(span[1])) : frame.width;
  if (to <= from) return;
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1.5;
  for (const lane of frame.lanes) {
    ctx.beginPath();
    for (let x = Math.floor(from); x <= Math.ceil(to); x += 1) {
      const y = lane.top + 2 + (1 - Math.max(0, Math.min(1, gainAt(frame.timeAt(x))))) * (lane.height - 4);
      if (x === Math.floor(from)) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

export { drawLabel };
