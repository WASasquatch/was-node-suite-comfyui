/**
 * The mix Audio Mixer made, with a row per track: its name, whether it played, its fader, its
 * pan and its peak level after both.
 */

import { app } from "../../scripts/app.js";
import { formatLevel, mountAudioPanel } from "./interface/audio_view.js";
import { themeVar } from "./interface/theme.js";
import { el } from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.AudioMixer";
const NODE_ID = "WASAudioMixer";

// The quietest peak the row meters reach, in dBFS.
const METER_FLOOR = -48;
// Tracks one mixer sums at most, the node's own limit.
const MAX_TRACKS = 64;

/** A pan position as the row prints it. */
function panText(pan) {
  const value = Number(pan) || 0;
  if (Math.abs(value) < 0.005) return "C";
  return `${value < 0 ? "L" : "R"}${Math.round(Math.abs(value) * 100)}`;
}

function trackTable() {
  const element = el("div", "flex:0 1 auto;max-height:40%;overflow:auto;display:grid;"
    + "grid-template-columns:minmax(0,1fr) auto auto auto 64px;column-gap:8px;row-gap:2px;"
    + `align-items:center;font-variant-numeric:tabular-nums;color:${themeVar("fg")}`);
  const update = (report) => {
    const rows = report?.mixer?.tracks;
    element.replaceChildren();
    element.style.display = Array.isArray(rows) && rows.length ? "grid" : "none";
    if (!Array.isArray(rows)) return;
    for (const row of rows.slice(0, MAX_TRACKS)) {
      const playing = row.state === "playing";
      const tone = playing ? themeVar("fg") : themeVar("fgMuted");
      const name = el("span", `overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:${tone}`,
        String(row.name ?? ""));
      name.title = `${row.name}: ${row.state}, starts at ${Number(row.start).toFixed(2)} s, `
        + `${Number(row.duration).toFixed(2)} s long`;
      const state = el("span", `color:${playing ? themeVar("success") : themeVar("warning")}`,
        playing ? "on" : row.state === "muted" ? "muted" : "off");
      const gain = el("span", `color:${tone};text-align:right`, `${Number(row.gain_db) >= 0 ? "+" : ""}${Number(row.gain_db).toFixed(1)} dB`);
      const pan = el("span", `color:${tone};text-align:right;min-width:26px`, panText(row.pan));
      const meter = el("div", `position:relative;height:6px;border-radius:3px;background:${themeVar("bgDark")};overflow:hidden`);
      const peak = typeof row.peak_db === "number" ? row.peak_db : null;
      const share = peak === null ? 0 : Math.max(0, Math.min(1, (peak - METER_FLOOR) / -METER_FLOOR));
      const colour = peak !== null && peak > -3 ? themeVar("error") : peak !== null && peak > -12 ? themeVar("warning") : themeVar("success");
      meter.append(el("div", `position:absolute;left:0;top:0;bottom:0;width:${(share * 100).toFixed(1)}%;background:${colour}`));
      meter.title = `peak ${formatLevel(peak)}`;
      element.append(name, state, gain, pan, meter);
    }
  };
  return { element, update };
}

function mixerFigures(report) {
  const rows = report?.mixer?.tracks;
  if (!Array.isArray(rows)) return undefined;
  const playing = rows.filter((row) => row.state === "playing").length;
  const master = Number(report?.mixer?.master_db) || 0;
  return `${rows.length} track${rows.length === 1 ? "" : "s"}, ${playing} playing · master `
    + `${master >= 0 ? "+" : ""}${master.toFixed(1)} dB · peak ${formatLevel(report?.out?.peak_db)}`;
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_audio_mixer_wrapped) return;
    proto.__was_audio_mixer_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        mountAudioPanel(this, {
          sides: [{ key: "out", label: "Mix" }],
          initial: "out",
          height: 230,
          figures: mixerFigures,
          extra: () => trackTable(),
        });
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the mixer panel:`, error);
      }
      return result;
    };
  },
});
