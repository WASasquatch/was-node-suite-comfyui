/**
 * The waveform editor on Audio Editor: the kept part between two edges, the cuts and the end
 * fades, each dragged into place on the source.
 *
 * Play sounds the edit before the node runs.
 */

import { app } from "../../scripts/app.js";
import {
  curveGain,
  drawEnvelope,
  findWidget,
  mountAudioPanel,
  readNumber,
  roundTime,
  shadeSpan,
} from "./interface/audio_view.js";
import { appendEntry, listEntries, parseRegion, replaceEntry, withoutEntry } from "./interface/audio_text.js";
import { chainWidgetCallback } from "./interface/widget.js";
import { button, setDisabled } from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.AudioEditor";
const NODE_ID = "WASAudioEditor";

// Pixels an edge or a handle can be grabbed from.
const GRAB = 7;

// Widgets whose hand edits move something the editor draws.
const WATCHED = ["start", "end", "cuts", "fade_in", "fade_out", "curve"];

/** The cuts widget's entries, each with the region it names or null. */
function cutEntries(text) {
  return listEntries(text, parseRegion).map((entry) => ({ ...entry, region: entry.value }));
}

/** The editor's state read from its widgets, with whatever a drag is holding. */
function settings(panel, held) {
  const node = panel.node;
  const total = panel.clipOf("in")?.duration ?? 0;
  const start = held?.start ?? readNumber(node, "start");
  const endValue = held?.end ?? readNumber(node, "end");
  const end = endValue > 0 ? Math.min(total, endValue) : total;
  const entries = cutEntries(findWidget(node, "cuts")?.value ?? "");
  return {
    total,
    start: Math.min(start, total),
    end,
    fadeIn: held?.fade_in ?? readNumber(node, "fade_in"),
    fadeOut: held?.fade_out ?? readNumber(node, "fade_out"),
    curve: String(findWidget(node, "curve")?.value ?? "equal power"),
    entries,
    regions: entries.filter((entry) => entry.region).map((entry) => entry.region),
    unreadable: entries.filter((entry) => !entry.region).length,
  };
}

/** The kept pieces between the edges and around the cuts, in seconds of the source. */
function keptPieces(state) {
  const pieces = [];
  let cursor = state.start;
  const sorted = [...state.regions].sort((a, b) => a[0] - b[0]);
  for (const [from, to] of sorted) {
    const a = Math.max(state.start, from);
    const b = Math.min(state.end, to);
    if (b <= a) continue;
    if (a > cursor) pieces.push({ from: cursor, to: a });
    cursor = Math.max(cursor, b);
  }
  if (cursor < state.end) pieces.push({ from: cursor, to: state.end });
  return pieces;
}

function editorTool(panel) {
  let drag = null;
  let held = null;
  let chosen = -1;

  const read = () => settings(panel, held);
  const onSource = () => panel.side() === "in";
  const targets = (at) => {
    const state = read();
    const lane = panel.lanes()[0];
    const top = lane ? lane.top : 0;
    const found = [];
    found.push({ kind: "fade_in", x: panel.xAt(state.start + state.fadeIn), near: Math.abs(at.y - (top + 5)) <= GRAB * 2 });
    found.push({ kind: "fade_out", x: panel.xAt(state.end - state.fadeOut), near: Math.abs(at.y - (top + 5)) <= GRAB * 2 });
    found.push({ kind: "start", x: panel.xAt(state.start), near: true });
    found.push({ kind: "end", x: panel.xAt(state.end), near: true });
    state.entries.forEach((entry, index) => {
      if (!entry.region) return;
      found.push({ kind: "cut_from", index, x: panel.xAt(entry.region[0]), near: true });
      found.push({ kind: "cut_to", index, x: panel.xAt(entry.region[1]), near: true });
    });
    return found.filter((item) => item.near && Math.abs(item.x - at.x) <= GRAB);
  };

  const writeCuts = (text) => panel.write("cuts", text);

  const rewrite = (index, region) => {
    const value = String(findWidget(panel.node, "cuts")?.value ?? "");
    const entry = cutEntries(value)[index];
    if (entry) writeCuts(replaceEntry(value, entry, `${region[0].toFixed(3)}-${region[1].toFixed(3)}`));
  };

  const tool = {
    settings: () => read(),
    chosen: () => chosen,
    cursor(at) {
      if (!onSource()) return "";
      const hit = targets(at)[0];
      return hit ? "ew-resize" : "";
    },
    pointerDown(event, at) {
      if (!onSource()) return false;
      const hit = targets(at)[0];
      const state = read();
      if (!hit) {
        const index = state.entries.findIndex((entry) => entry.region && at.time >= entry.region[0] && at.time <= entry.region[1]);
        if (index >= 0) {
          chosen = index;
          panel.setSelection(null);
          panel.repaint();
          return true;
        }
        chosen = -1;
        return false;
      }
      drag = hit;
      held = {
        start: state.start,
        end: readNumber(panel.node, "end"),
        fade_in: state.fadeIn,
        fade_out: state.fadeOut,
      };
      if (hit.kind.startsWith("cut")) {
        chosen = hit.index;
        held.region = [...state.entries[hit.index].region];
      }
      return true;
    },
    pointerMove(event, at) {
      if (!drag || !held) return;
      const state = read();
      const time = Math.max(0, Math.min(state.total, at.time));
      if (drag.kind === "start") held.start = roundTime(Math.min(time, state.end - 0.001));
      else if (drag.kind === "end") held.end = roundTime(Math.max(time, state.start + 0.001));
      else if (drag.kind === "fade_in") held.fade_in = roundTime(Math.max(0, time - state.start));
      else if (drag.kind === "fade_out") held.fade_out = roundTime(Math.max(0, state.end - time));
      else if (drag.kind === "cut_from") held.region[0] = roundTime(Math.min(time, held.region[1]));
      else if (drag.kind === "cut_to") held.region[1] = roundTime(Math.max(time, held.region[0]));
      panel.repaint();
    },
    pointerUp() {
      if (drag && held) {
        const total = read().total;
        if (drag.kind === "start") panel.write("start", held.start);
        else if (drag.kind === "end") panel.write("end", held.end >= total - 0.0005 ? 0 : held.end);
        else if (drag.kind === "fade_in") panel.write("fade_in", held.fade_in);
        else if (drag.kind === "fade_out") panel.write("fade_out", held.fade_out);
        else if (held.region) rewrite(drag.index, held.region);
      }
      drag = null;
      held = null;
      panel.repaint();
    },
    heldRegion: () => (drag?.kind?.startsWith("cut") ? { index: drag.index, region: held?.region } : null),
    trimToSelection() {
      const range = panel.selection();
      if (!range) return;
      const total = read().total;
      panel.write("start", roundTime(range[0]));
      panel.write("end", range[1] >= total - 0.0005 ? 0 : roundTime(range[1]));
      panel.setSelection(null);
    },
    cutSelection() {
      const range = panel.selection();
      if (!range) return false;
      const value = String(findWidget(panel.node, "cuts")?.value ?? "");
      writeCuts(appendEntry(value, `${roundTime(range[0]).toFixed(3)}-${roundTime(range[1]).toFixed(3)}`));
      panel.setSelection(null);
      return true;
    },
    deleteChosen() {
      const value = String(findWidget(panel.node, "cuts")?.value ?? "");
      const entry = cutEntries(value)[chosen];
      if (!entry) return false;
      writeCuts(withoutEntry(value, entry));
      chosen = -1;
      return true;
    },
    reset() {
      panel.write("start", 0);
      panel.write("end", 0);
      panel.write("cuts", "");
      panel.write("fade_in", 0);
      panel.write("fade_out", 0);
      chosen = -1;
      panel.setSelection(null);
    },
    key(event) {
      if (event.key === "Delete" || event.key === "Backspace") {
        if (!tool.cutSelection()) tool.deleteChosen();
        return true;
      }
      if (event.key === "t" || event.key === "T") {
        tool.trimToSelection();
        return true;
      }
      return false;
    },
    selectionChanged() {
      tool.onSelection?.();
    },
  };
  return tool;
}

function editorLayer(ctx, frame, panel) {
  const tool = panel.tool;
  const state = tool?.settings?.();
  if (!state || panel.side() !== "in") return;
  const moving = tool.heldRegion();
  shadeSpan(ctx, frame, 0, state.start, frame.theme.bg, 0.7);
  shadeSpan(ctx, frame, state.end, state.total, frame.theme.bg, 0.7);
  state.entries.forEach((entry, index) => {
    const region = moving && moving.index === index ? moving.region : entry.region;
    if (!region) return;
    shadeSpan(ctx, frame, region[0], region[1], frame.theme.error, index === tool.chosen() ? 0.42 : 0.26);
    ctx.fillStyle = frame.theme.error;
    for (const edge of region) ctx.fillRect(Math.round(frame.xAt(edge)), frame.top, 1, frame.height - frame.top);
  });
  drawEnvelope(ctx, frame, (time) => {
    let gain = 1;
    if (state.fadeIn > 0 && time < state.start + state.fadeIn) gain *= curveGain(state.curve, (time - state.start) / state.fadeIn);
    if (state.fadeOut > 0 && time > state.end - state.fadeOut) gain *= curveGain(state.curve, (state.end - time) / state.fadeOut);
    return gain;
  }, frame.theme.warning, [state.start, state.end]);
  ctx.fillStyle = frame.theme.warning;
  for (const edge of [state.start, state.end]) {
    const x = Math.round(frame.xAt(edge));
    ctx.fillRect(x - 1, frame.top, 3, frame.height - frame.top);
    ctx.fillRect(x - 4, frame.height - 10, 9, 10);
  }
  const top = frame.lanes[0]?.top ?? frame.top;
  for (const at of [state.start + state.fadeIn, state.end - state.fadeOut]) {
    ctx.fillRect(Math.round(frame.xAt(at)) - 4, top + 1, 8, 8);
  }
}

function editorFigures(report, side, panel) {
  const state = panel.tool?.settings?.();
  if (!state || !state.total) return undefined;
  const kept = keptPieces(state).reduce((sum, piece) => sum + (piece.to - piece.from), 0);
  const cuts = state.regions.length;
  const unread = state.unreadable ? ` · ${state.unreadable} unreadable cut${state.unreadable === 1 ? "" : "s"}` : "";
  return `keeps ${kept.toFixed(3)} s of ${state.total.toFixed(3)} s · ${cuts} cut${cuts === 1 ? "" : "s"}${unread}`;
}

function editorAudition(panel) {
  if (panel.side() !== "in" || panel.selection()) return null;
  const state = panel.tool?.settings?.();
  if (!state) return null;
  const pieces = keptPieces(state);
  if (!pieces.length) return null;
  pieces[0].fadeIn = state.fadeIn;
  pieces[pieces.length - 1].fadeOut = state.fadeOut;
  return pieces;
}

function editorControls(bar, panel) {
  const small = "padding:1px 8px;font-size:11px;line-height:16px;border-radius:4px";
  const trim = button("Trim", () => panel.tool.trimToSelection(), {
    kind: "tool", logName: EXT_NAME, hint: "Keep only the selection (T)",
  });
  const cut = button("Cut", () => panel.tool.cutSelection(), {
    kind: "tool", logName: EXT_NAME, hint: "Take the selection out (Delete)",
  });
  const reset = button("Reset", () => panel.tool.reset(), {
    kind: "tool", logName: EXT_NAME, hint: "Clear the trim, the cuts and the fades",
  });
  for (const item of [trim, cut, reset]) item.style.cssText += `;${small}`;
  const update = () => {
    const none = !panel.selection();
    setDisabled(trim, none);
    setDisabled(cut, none);
  };
  panel.tool.onSelection = update;
  update();
  bar.append(trim, cut, reset);
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_audio_editor_wrapped) return;
    proto.__was_audio_editor_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const built = mountAudioPanel(this, {
          sides: [{ key: "in", label: "Source" }, { key: "out", label: "Edited" }],
          initial: "in",
          select: true,
          height: 220,
          tool: editorTool,
          layers: [editorLayer],
          figures: editorFigures,
          audition: editorAudition,
          controls: editorControls,
        });
        if (built) {
          for (const name of WATCHED) chainWidgetCallback(this, name, () => built.panel.repaint(), EXT_NAME);
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the editor:`, error);
      }
      return result;
    };
  },
});
