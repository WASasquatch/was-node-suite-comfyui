/**
 * The clips Audio Split made, drawn as numbered spans over the source.
 *
 * In times mode the cut points are the widget's own: a click adds one, a drag moves it and a
 * double click removes it.
 */

import { app } from "../../scripts/app.js";
import { appendEntry, listEntries, parseTime, replaceEntry, withoutEntry } from "./interface/audio_text.js";
import { drawLabel, drawMarkers, findWidget, mountAudioPanel, roundTime, shadeSpan } from "./interface/audio_view.js";
import { chainWidgetCallback } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.AudioSplit";
const NODE_ID = "WASAudioSplit";

// Pixels a cut point can be grabbed from.
const GRAB = 6;

// Pixels a press may move and still read as a click that places a cut.
const CLICK_SLOP = 3;

const timesMode = (panel) => String(findWidget(panel.node, "mode")?.value ?? "") === "times";
const timeEntries = (panel) => listEntries(findWidget(panel.node, "times")?.value ?? "", parseTime);

function splitTool(panel) {
  let drag = null;
  const near = (at) => timeEntries(panel).findIndex((entry) => entry.value !== null
    && Math.abs(panel.xAt(entry.value) - at.x) <= GRAB);
  const write = (text) => panel.write("times", text);
  return {
    dragging: () => drag,
    cursor: (at) => (timesMode(panel) ? (near(at) >= 0 ? "ew-resize" : "copy") : ""),
    pointerDown(event, at) {
      if (!timesMode(panel)) return false;
      const index = near(at);
      drag = { index, start: at, time: index >= 0 ? timeEntries(panel)[index].value : at.time, moved: false };
      return true;
    },
    pointerMove(event, at) {
      if (!drag) return;
      if (Math.abs(at.x - drag.start.x) > CLICK_SLOP) drag.moved = true;
      if (drag.index >= 0 && drag.moved) {
        drag.time = roundTime(Math.min(panel.clip()?.duration ?? at.time, at.time));
        panel.repaint();
      }
    },
    pointerUp() {
      const ended = drag;
      drag = null;
      if (!ended) return;
      const value = String(findWidget(panel.node, "times")?.value ?? "");
      if (ended.index < 0 && !ended.moved) write(appendEntry(value, roundTime(ended.start.time).toFixed(3)));
      else if (ended.index >= 0 && ended.moved) {
        const entry = timeEntries(panel)[ended.index];
        if (entry) write(replaceEntry(value, entry, ended.time.toFixed(3)));
      } else panel.seek(ended.start.time);
      panel.repaint();
    },
    doubleClick(event, at) {
      if (!timesMode(panel)) return false;
      const index = near(at);
      if (index < 0) return true;
      const value = String(findWidget(panel.node, "times")?.value ?? "");
      write(withoutEntry(value, timeEntries(panel)[index]));
      return true;
    },
  };
}

function splitLayer(ctx, frame, panel) {
  const total = panel.clip()?.duration ?? 0;
  if (timesMode(panel)) {
    const held = panel.tool?.dragging?.();
    const times = timeEntries(panel).map((entry, index) => (held && held.index === index && held.moved ? held.time : entry.value))
      .filter((time) => time !== null && time > 0 && time < total)
      .sort((a, b) => a - b);
    const edges = [0, ...times, total];
    edges.slice(0, -1).forEach((from, index) => {
      if (index % 2) shadeSpan(ctx, frame, from, edges[index + 1], frame.theme.accent, 0.1);
    });
    drawMarkers(ctx, frame, times, frame.theme.warning);
    edges.slice(0, -1).forEach((from, index) => drawLabel(ctx, frame, String(index + 1), frame.xAt(from) + 4, frame.top + 9));
    return;
  }
  const spans = panel.report()?.split?.spans ?? [];
  let cursor = 0;
  spans.forEach(([from, to], index) => {
    if (from > cursor) shadeSpan(ctx, frame, cursor, from, frame.theme.bg, 0.65);
    if (index % 2) shadeSpan(ctx, frame, from, to, frame.theme.accent, 0.1);
    cursor = to;
  });
  if (spans.length && cursor < total) shadeSpan(ctx, frame, cursor, total, frame.theme.bg, 0.65);
  drawMarkers(ctx, frame, spans.flatMap(([from, to]) => [from, to]).filter((time) => time > 0 && time < total),
    frame.theme.warning);
  spans.forEach(([from], index) => drawLabel(ctx, frame, String(index + 1), frame.xAt(from) + 4, frame.top + 9));
}

function splitFigures(report, side, panel) {
  if (timesMode(panel)) {
    const entries = timeEntries(panel);
    const unread = entries.filter((entry) => entry.value === null).length;
    const cuts = entries.length - unread;
    return `${cuts} cut${cuts === 1 ? "" : "s"}: click to add, drag to move, double-click to remove`
      + (unread ? ` · ${unread} unreadable` : "");
  }
  const spans = report?.split?.spans;
  return Array.isArray(spans) ? `${spans.length} clip${spans.length === 1 ? "" : "s"}` : undefined;
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_audio_split_wrapped) return;
    proto.__was_audio_split_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const built = mountAudioPanel(this, {
          sides: [{ key: "in", label: "Source" }],
          initial: "in",
          tool: splitTool,
          layers: [splitLayer],
          figures: splitFigures,
        });
        if (built) {
          const node = this;
          // The times box is made and unmade as the mode changes, so it is followed again each time.
          const follow = () => {
            const box = findWidget(node, "times");
            if (box && !box.__was_split_followed) {
              box.__was_split_followed = true;
              chainWidgetCallback(node, box.name, () => built.panel.repaint(), EXT_NAME);
            }
            built.panel.repaint();
          };
          const mode = findWidget(node, "mode");
          if (mode) chainWidgetCallback(node, mode.name, () => setTimeout(follow, 0), EXT_NAME);
          follow();
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the split panel:`, error);
      }
      return result;
    };
  },
});
